import os
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

from logging_config import logger
from modeling.batch_processing import (
    build_ray_dataset_from_directory,
    run_object_detector_against_ray_dataset,
    convert_detections_to_coco_format,
    build_naive_detector_config_from_coco_categories_metadata,
    filter_dataset_by_path,
)


def download_coco_data(
    split: str, to_dir: str = "/directai_fastapi/.cache/coco"
) -> None:
    """Download COCO data for the given split if not present."""
    # Define your paths and URLs for COCO datasets
    if split not in ["train", "val", "test"]:
        raise ValueError(f"Invalid split: {split}. Expected 'train', 'val', or 'test'.")

    annotation_url = (
        f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    )
    images_url = f"http://images.cocodataset.org/zips/{split}2017.zip"

    annotation_zip_file = os.path.join(to_dir, f"annotations_trainval2017.zip")
    images_zip_file = os.path.join(to_dir, f"{split}2017.zip")

    # Check if files exist, download if not
    if not os.path.exists(annotation_zip_file) or not os.path.exists(images_zip_file):
        os.makedirs(to_dir, exist_ok=True)
        # Download annotations
        os.system(f"wget {annotation_url} -O {annotation_zip_file}")
        os.system(f"unzip -o {annotation_zip_file} -d {to_dir}")
        # Download images
        os.system(f"wget {images_url} -O {images_zip_file}")
        os.system(f"unzip -o {images_zip_file} -d {to_dir}")


if __name__ == "__main__":
    coco_base_dir = "/directai_fastapi/.cache/coco"
    split = "val"

    logger.info(f"Downloading COCO data for split: {split}")
    download_coco_data(split, coco_base_dir)

    images_dir = os.path.join(coco_base_dir, f"{split}2017")
    annotations_path = os.path.join(
        coco_base_dir, "annotations", f"instances_{split}2017.json"
    )
    predictions_path = os.path.join(coco_base_dir, f"predictions_{split}2017.json")

    # Download COCO data
    download_coco_data(split, to_dir=coco_base_dir)

    # Load COCO ground truth
    coco_gt = COCO(annotations_path)

    img_ids = coco_gt.getImgIds()
    # img_ids = img_ids[:10]  # Limit to 100 images for testing

    # for the COCO gt object, the cats and imgs attributes are dictionaries with id as key
    # instead of the list of dictionaries that is the supposed format
    # so we convert them to a list of dictionaries
    categories_metadata = [c for c in coco_gt.cats.values()]
    images_metadata = [i for i in coco_gt.imgs.values()]

    paths_set = set([coco_gt.imgs[id]["file_name"] for id in img_ids])

    # Build naive detector config from COCO categories metadata
    labels, inc_sub_labels_dict = (
        build_naive_detector_config_from_coco_categories_metadata(categories_metadata)
    )
    label_conf_thres = {name: 0.001 for name in labels}

    # print(labels)
    # print(categories_metadata)
    # print(coco_gt.cats)
    # print(img_ids)

    # Run object detector against COCO dataset
    ray_dataset = build_ray_dataset_from_directory(
        images_dir,
        with_subdirs_as_labels=False,
        remove_extension=False,
    )
    # ray_dataset = ray_dataset.limit(100)  # Limit to 100 images for testing
    ray_dataset = filter_dataset_by_path(ray_dataset, paths_set)
    predictions = run_object_detector_against_ray_dataset(
        ray_dataset,
        batch_size=64,
        labels=labels,
        inc_sub_labels_dict=inc_sub_labels_dict,
        label_conf_thres=label_conf_thres,
        nms_thre=0.9,
    )
    _image_metadata, _categories_metadata, coco_format_predictions = (
        convert_detections_to_coco_format(
            predictions,
            labels,
            images_metadata,
            categories_metadata,
        )
    )

    # Save predictions in COCO format
    with open(predictions_path, "w") as f:
        json.dump(coco_format_predictions, f)

    # Load COCO predictions
    coco_dt = coco_gt.loadRes(predictions_path)

    # Evaluate COCO predictions
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
