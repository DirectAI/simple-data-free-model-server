import ray
from ray.data.datasource import PartitionStyle
from ray.data.datasource.partitioning import Partitioning
import numpy as np
from functools import partial
import torch
from torchvision.transforms import v2  # type: ignore[import-untyped]
from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]
import os

from modeling.image_classifier import ZeroShotImageClassifierWithFeedback


def dir_name_to_label(dir_name: str) -> str:
    return dir_name.replace("_", " ")


def truncate_full_path(
    row: dict[str, np.ndarray | str], last_n: int = 1, remove_extension: bool = True
) -> dict[str, np.ndarray | str]:
    assert isinstance(row["path"], str)
    last_n_path = "/".join(row["path"].split("/")[-last_n:])
    if remove_extension:
        last_n_path = last_n_path.rsplit(".")[0]
    row["path"] = last_n_path
    return row


def build_ray_dataset_from_directory(
    root: str,
    with_subdirs_as_labels: bool = True,
) -> ray.data.Dataset:
    # we expect images to be stored directly in the root directory
    # unless with_subdirs_as_labels is set to True
    # in which case we expect images to be stored in subdirectories
    # the sames of which will be used as labels for the images
    if with_subdirs_as_labels:
        partitioning = Partitioning(
            PartitionStyle.DIRECTORY,
            field_names=[
                "label",
            ],
            base_dir=root,
        )
    else:
        partitioning = None

    ds = ray.data.read_images(
        root, mode="RGB", partitioning=partitioning, include_paths=True
    )

    ds = ds.map(partial(truncate_full_path, last_n=2 if with_subdirs_as_labels else 1))

    return ds


def get_labels_from_directory(
    root: str,
) -> list[str]:
    subdirs = [
        name for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))
    ]
    labels = [name for name in subdirs if not name.startswith(".")]
    return labels


def preprocess_image_for_classifier(
    row: dict[str, np.ndarray], image_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    image = row["image"]
    image = v2.functional.to_image(image)
    image = v2.functional.to_dtype(image, torch.float32, scale=False)
    image = v2.functional.resize(
        image, image_size, interpolation=InterpolationMode.BICUBIC
    )
    # NOTE: if we do this without modifying the original, we may be able to speed thigns up
    # see https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html#ray.data.Dataset.map_batches
    assert isinstance(image, torch.Tensor)
    row["image"] = image.numpy()
    return row


class RayDataImageClassifier:
    def __init__(
        self,
        labels: list[str],
        inc_sub_labels_dict: dict[str, list[str]],
        exc_sub_labels_dict: dict[str, list[str]] | None = None,
        augment_examples: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ZeroShotImageClassifierWithFeedback(device=self.device)

        self.labels = labels
        self.inc_sub_labels_dict = inc_sub_labels_dict
        self.exc_sub_labels_dict = exc_sub_labels_dict
        self.augment_examples = augment_examples

        self.fill_cache()

    def fill_cache(self) -> None:
        # compute embeddings for all prompts and warm up autocasting
        for _ in range(4):
            rand_image = np.random.rand(1, 3, 224, 224).astype(np.float32)
            batch: dict[str, np.ndarray] = {
                "image": rand_image,
                "label": np.array(
                    [
                        "dummy",
                    ]
                ),
            }
            self(batch)

    def __call__(
        self,
        batch: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        images = torch.from_numpy(batch.pop("image")).to(self.device)

        with torch.inference_mode(), torch.autocast(str(self.model.device)):
            raw_scores = self.model(
                images,
                labels=self.labels,
                inc_sub_labels_dict=self.inc_sub_labels_dict,
                exc_sub_labels_dict=self.exc_sub_labels_dict,
                augment_examples=self.augment_examples,
            )
            scores = torch.nn.functional.softmax(raw_scores / 0.07, dim=1)
            ind = scores.argmax(dim=1).cpu().numpy()
            pred = np.array([labels[i] for i in ind])

        batch["scores"] = scores.cpu().numpy()
        batch["pred"] = pred

        if "label" in batch:
            is_correct = np.array(
                [p == l for p, l in zip(pred, batch["label"])], dtype=np.float32
            )

            batch["is_correct"] = is_correct

        return batch


def run_image_classifier_against_ray_dataset(
    ds: ray.data.Dataset,
    batch_size: int,
    labels: list[str],
    inc_sub_labels_dict: dict[str, list[str]],
    exc_sub_labels_dict: dict[str, list[str]] | None = None,
    augment_examples: bool = True,
    concurrency: int | None = None,
    num_gpus: int | None = None,
) -> ray.data.Dataset:
    if num_gpus is None:
        # the number of GPUs to reserve for each actor
        num_gpus = 1

    if concurrency is None:
        # the number of actors to create
        # set to the number of available GPUs
        concurrency = torch.cuda.device_count()

    # we are going to assume that the image classifier wants 224x224 images
    preprocessed_ds = ds.map(
        partial(preprocess_image_for_classifier, image_size=(224, 224))
    )

    predictions = preprocessed_ds.map_batches(
        RayDataImageClassifier,
        batch_size=batch_size,
        concurrency=concurrency,
        num_gpus=num_gpus,
        fn_constructor_kwargs={
            "labels": labels,
            "inc_sub_labels_dict": inc_sub_labels_dict,
            "exc_sub_labels_dict": exc_sub_labels_dict,
            "augment_examples": augment_examples,
        },
    )

    return predictions


def write_predictions_to_csv(
    predictions: ray.data.Dataset,
    output_file: str,
) -> None:
    # Ray data has a built-in method to write to CSV
    # but it does it by partition into multiple files
    # we're just going to write everything to a single file

    iterator = iter(predictions.iter_rows())

    with open(output_file, "w") as file:
        first_row = next(iterator)
        has_gt = "label" in first_row
        if has_gt:
            file.write("path,label,pred,is_correct\n")
        else:
            file.write("path,pred\n")

        def write_row(row: dict[str, np.ndarray]) -> None:
            if has_gt:
                file.write(
                    f"{row['path']},{row['label']},{row['pred']},{row['is_correct']}\n"
                )
            else:
                file.write(f"{row['path']},{row['pred']}\n")

        write_row(first_row)

        for row in iterator:
            write_row(row)


if __name__ == "__main__":
    # first we load the pytorch dataset and save it to disk
    # we're not going to actually use it, but we want to make sure it's available
    # so we can assemble the dataset from the directory
    from torchvision.datasets import Food101  # type: ignore[import-untyped]

    food101_root_dir = "/directai_fastapi/.cache/Food101"
    _ = Food101(root=food101_root_dir, download=True)

    food101_images_dir = os.path.join(food101_root_dir, "food-101", "images")

    labels = get_labels_from_directory(food101_images_dir)
    ds = build_ray_dataset_from_directory(
        food101_images_dir, with_subdirs_as_labels=True
    )

    # NOTE: the ds is not split into training and validation sets
    # for this specific dataset, we can split it according to the provided test.txt and train.txt files
    split = "test"
    split_file = os.path.join(food101_root_dir, "food-101", "meta", f"{split}.txt")
    with open(split_file, "r") as file:
        split_list = file.read().splitlines()
    split_set = set(split_list)
    # we then filter rows based on whether the path is in the split set
    ds = ds.filter(lambda x: x["path"] in split_set)

    inc_sub_labels_dict = {
        label: [
            dir_name_to_label(label),
        ]
        for label in labels
    }
    augment_examples = True

    predictions = run_image_classifier_against_ray_dataset(
        ds,
        batch_size=1024,
        labels=labels,
        inc_sub_labels_dict=inc_sub_labels_dict,
        augment_examples=augment_examples,
    )

    # we compute the mean of the is_correct column to get the accuracy
    # since Ray data pipelines are lazy, this triggers actually running the pipeline
    # accuracy = predictions.mean("is_correct")

    # print(f"Accuracy on {split} set: {accuracy}")

    write_predictions_to_csv(
        predictions,
        os.path.join(food101_root_dir, "food-101", "meta", f"{split}_predictions.csv"),
    )
