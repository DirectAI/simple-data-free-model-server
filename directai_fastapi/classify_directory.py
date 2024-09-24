import fire  # type: ignore[import-untyped]

from modeling.batch_processing import (
    build_ray_dataset_from_directory,
    run_image_classifier_against_ray_dataset,
    write_classifications_to_csv,
    filter_dataset_by_path,
)
from pydantic_models import ClassifierDeploy
from logging_config import logger


def classify_directory(
    root: str,
    classifier_json_file: str,
    output_file: str | None = None,
    batch_size: int = 1024,
    concurrency: int | None = None,
    images_to_classify_fp: str | None = None,
    eval_only: bool = False,
) -> None:
    assert (
        output_file is not None or eval_only
    ), "Output file must be provided if not doing just an eval."
    if output_file is not None:
        assert output_file.endswith(".csv"), "Output file must be a CSV file."

    classifier_deploy = ClassifierDeploy.parse_file(classifier_json_file)
    classifier_config_dict = classifier_deploy.build_config_dict()

    ds = build_ray_dataset_from_directory(root)

    if images_to_classify_fp is not None:
        ds = filter_dataset_by_path(ds, paths_file=images_to_classify_fp)

    predictions = run_image_classifier_against_ray_dataset(
        ds,
        batch_size,
        classifier_config_dict["labels"],
        classifier_config_dict["inc_sub_labels_dict"],
        classifier_config_dict["exc_sub_labels_dict"],
        classifier_config_dict["augment_examples"],
        concurrency,
    )

    if eval_only:
        assert "label" in ds.columns(), "Dataset must have labels to evaluate."
        # there were labels in the dataset, so we can calculate accuracy
        accuracy = predictions.mean("is_correct")
        logger.info(f"Accuracy: {accuracy}")
    else:
        assert (
            output_file is not None
        ), "Output file must be provided if not doing just an eval."
        write_classifications_to_csv(predictions, output_file)


if __name__ == "__main__":
    fire.Fire(classify_directory)
