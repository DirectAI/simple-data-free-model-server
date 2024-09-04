import ray
from ray.data.datasource import PartitionStyle
from ray.data.datasource.partitioning import Partitioning
import numpy as np
from functools import partial
import torch
from torchvision.transforms import v2  # type: ignore[import-untyped]
from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]
import os
from typing import Any

from modeling.image_classifier import ZeroShotImageClassifierWithFeedback
from modeling.object_detector import ZeroShotObjectDetectorWithFeedback


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
    # the names of which will be used as labels for the images
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


def filter_dataset_by_path(
    dataset: ray.data.Dataset,
    paths_set: set[str] | None = None,
    paths_file: str | None = None,
) -> ray.data.Dataset:
    if paths_set is None:
        assert paths_file is not None
        with open(paths_file, "r") as file:
            paths_set = set(file.read().splitlines())

    return dataset.filter(lambda x: x["path"] in paths_set)


def filter_dataset_by_label(
    dataset: ray.data.Dataset,
    labels_set: set[str] | None = None,
    labels_file: str | None = None,
) -> ray.data.Dataset:
    if labels_set is None:
        assert labels_file is not None
        with open(labels_file, "r") as file:
            labels_set = set(file.read().splitlines())

    return dataset.filter(lambda x: x["label"] in labels_set)


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


def preprocess_image_for_detector(
    row: dict[str, np.ndarray], image_size: tuple[int, int]
) -> dict[str, np.ndarray]:
    image = row["image"]
    image_initial_size = image.shape[:2]

    padded_tensor = torch.ones((3, *image_size), dtype=torch.float32)

    r = min(
        image_size[0] / image_initial_size[0], image_size[1] / image_initial_size[1]
    )
    target_size = (int(image_initial_size[0] * r), int(image_initial_size[1] * r))

    image = v2.functional.to_image(image)
    image = v2.functional.to_dtype(image, torch.float32, scale=False)
    image = v2.functional.resize(
        image, target_size, interpolation=InterpolationMode.BICUBIC
    )

    assert isinstance(padded_tensor, torch.Tensor)
    assert isinstance(image, torch.Tensor)
    padded_tensor[:, : target_size[0], : target_size[1]] = image

    row["image"] = padded_tensor.numpy()
    row["image_scale_ratio"] = np.array(
        [
            r,
        ]
    )
    row["image_initial_size"] = np.array(image_initial_size)

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

    def run_model(self, images: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(str(self.model.device)):
            raw_scores = self.model(
                images,
                labels=self.labels,
                inc_sub_labels_dict=self.inc_sub_labels_dict,
                exc_sub_labels_dict=self.exc_sub_labels_dict,
                augment_examples=self.augment_examples,
            )
            scores = torch.nn.functional.softmax(raw_scores / 0.07, dim=1)
        return scores

    def fill_cache(self) -> None:
        # compute embeddings for all prompts and warm up autocasting
        for _ in range(4):
            rand_image = torch.rand((1, 3, 224, 224), device=self.device)
            _ = self.run_model(rand_image)

    def __call__(
        self,
        batch: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        images = torch.from_numpy(batch.pop("image")).to(self.device)

        scores = self.run_model(images)

        ind = scores.argmax(dim=1).cpu().numpy()
        pred = np.array([self.labels[i] for i in ind])

        batch["scores"] = scores.cpu().numpy()
        batch["pred"] = pred

        if "label" in batch:
            is_correct = np.array(
                [p == l for p, l in zip(pred, batch["label"])], dtype=np.float32
            )

            batch["is_correct"] = is_correct

        return batch


class RayDataObjectDetector:
    def __init__(
        self,
        labels: list[str],
        inc_sub_labels_dict: dict[str, list[str]],
        exc_sub_labels_dict: dict[str, list[str]] | None = None,
        label_conf_thres: dict[str, float] | None = None,
        augment_examples: bool = True,
        nms_thre: float = 0.4,
        run_class_agnostic_nms: bool = True,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ZeroShotObjectDetectorWithFeedback(device=self.device)

        self.labels = labels
        self.inc_sub_labels_dict = inc_sub_labels_dict
        self.exc_sub_labels_dict = exc_sub_labels_dict
        self.label_conf_thres = label_conf_thres
        self.augment_examples = augment_examples
        self.nms_thre = nms_thre
        self.run_class_agnostic_nms = run_class_agnostic_nms

        self.fill_cache()

    def run_model(
        self, images: torch.Tensor, image_scale_ratios: torch.Tensor
    ) -> list[list[torch.Tensor]]:
        with torch.inference_mode(), torch.autocast(str(self.model.device)):
            batched_predicted_boxes = self.model(
                images,
                labels=self.labels,
                inc_sub_labels_dict=self.inc_sub_labels_dict,
                exc_sub_labels_dict=self.exc_sub_labels_dict,
                label_conf_thres=self.label_conf_thres,
                augment_examples=self.augment_examples,
                nms_thre=self.nms_thre,
                run_class_agnostic_nms=self.run_class_agnostic_nms,
                image_scale_ratios=image_scale_ratios,
            )
        return batched_predicted_boxes

    def fill_cache(self) -> None:
        # compute embeddings for all prompts and warm up autocasting
        for _ in range(4):
            rand_image = torch.rand((1, 3, 1008, 1008), device=self.device)
            image_scale_ratio = torch.tensor(
                [
                    1,
                ],
                device=self.device,
            )
            _ = self.run_model(rand_image, image_scale_ratio)

    def __call__(
        self,
        batch: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        images = torch.from_numpy(batch.pop("image")).to(self.device)
        image_scale_ratios = torch.from_numpy(batch.pop("image_scale_ratio")).to(
            self.device
        )

        batched_predicted_boxes = self.run_model(images, image_scale_ratios)

        # TODO: this may not be representable via a numpy array
        batch["predicted_boxes"] = batched_predicted_boxes  # type: ignore[assignment]

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


def run_object_detector_against_ray_dataset(
    ds: ray.data.Dataset,
    batch_size: int,
    labels: list[str],
    inc_sub_labels_dict: dict[str, list[str]],
    exc_sub_labels_dict: dict[str, list[str]] | None = None,
    label_conf_thres: dict[str, float] | None = None,
    augment_examples: bool = True,
    nms_thre: float = 0.4,
    run_class_agnostic_nms: bool = True,
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

    # we are going to assume that the object detector wants 1008x1008 images
    preprocessed_ds = ds.map(
        partial(preprocess_image_for_detector, image_size=(1008, 1008))
    )

    predictions = preprocessed_ds.map_batches(
        RayDataObjectDetector,
        batch_size=batch_size,
        concurrency=concurrency,
        num_gpus=num_gpus,
        fn_constructor_kwargs={
            "labels": labels,
            "inc_sub_labels_dict": inc_sub_labels_dict,
            "exc_sub_labels_dict": exc_sub_labels_dict,
            "label_conf_thres": label_conf_thres,
            "augment_examples": augment_examples,
            "nms_thre": nms_thre,
            "run_class_agnostic_nms": run_class_agnostic_nms,
        },
    )

    return predictions


def write_classifications_to_csv(
    classifications: ray.data.Dataset,
    output_file: str,
) -> None:
    # Ray data has a built-in method to write to CSV
    # but it does it by partition into multiple files
    # we're just going to write everything to a single file

    iterator = iter(classifications.iter_rows())

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


def convert_detections_to_coco_format(
    detections: ray.data.Dataset,
    labels: list[str],
    image_metadata: list[dict[str, Any]] | None = None,
    categories_metadata: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # see https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html for the COCO format
    # if image_metadata or categories_metadata are not provided, we will try to infer them from the detections
    # NOTE: labels may not be redundant with categories_metadata, as the "id" field in categories_metadata may not be the same as the index of the label in labels

    need_to_infer_image_metadata = image_metadata is None
    need_to_infer_categories_metadata = categories_metadata is None

    if need_to_infer_image_metadata:
        image_metadata = []
    if need_to_infer_categories_metadata:
        categories_metadata = [
            {
                "id": i,
                "name": label,
                "supercategory": label,
            }
            for i, label in enumerate(labels)
        ]

    assert isinstance(categories_metadata, list)
    label_name_to_id = {
        category["name"]: category["id"] for category in categories_metadata
    }
    label_name_to_ind = {label: i for i, label in enumerate(labels)}
    label_ind_to_id = {
        label_name_to_ind[label]: label_name_to_id[label] for label in labels
    }

    assert isinstance(image_metadata, list)
    image_path_to_id = {data["file_name"]: data["id"] for data in image_metadata}

    annotations = []

    for row in detections.iter_rows():
        predicted_boxes = row["predicted_boxes"]

        if row["path"] not in image_path_to_id:
            image_id = len(image_metadata)
            image_h, image_w = row["image_initial_size"]
            image_metadata.append(
                {
                    "id": image_id,
                    "file_name": row["path"],
                    "height": image_h,
                    "width": image_w,
                    "date_captured": "",
                }
            )
        else:
            image_id = image_path_to_id[row["path"]]

        for i, predicted_boxes_in_class in enumerate(predicted_boxes):
            category_id = label_ind_to_id[i]

            for box in predicted_boxes_in_class:
                x_1, x_2, y_1, y_2, score = box

                x_1 = max(0, min(image_w, int(x_1)))
                x_2 = max(0, min(image_w, int(x_2)))
                y_1 = max(0, min(image_h, int(y_1)))
                y_2 = max(0, min(image_h, int(y_2)))

                annotations.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x_1, y_1, x_2 - x_1, y_2 - y_1],
                        "score": score,
                    }
                )

    return image_metadata, categories_metadata, annotations


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
    ds = filter_dataset_by_path(ds, paths_file=split_file)

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

    write_classifications_to_csv(
        predictions,
        os.path.join(food101_root_dir, "food-101", "meta", f"{split}_predictions.csv"),
    )
