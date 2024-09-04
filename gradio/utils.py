import json
import io
import requests
import gradio as gr  # type: ignore[import-untyped]

from typing import Tuple, Union, List, Optional
import numpy as np
from PIL import Image

from modeling import (
    SingleClassifierClass,
    SingleDetectorClass,
    ClassifierDeploy,
    DetectorDeploy,
    DualModelInterface,
    ModelType,
    deploy_classifier,
    deploy_detector,
    get_detector_results,
    get_classifier_results,
    append_flipped_bool_decorator,
)

from illustrating import display_bounding_boxes


@append_flipped_bool_decorator
def upload_file(
    file_bytes: bytes, models_state_val: DualModelInterface
) -> DualModelInterface:
    try:
        json_data = json.loads(file_bytes.decode("utf-8"))
        if models_state_val.current_model_type == ModelType.CLASSIFIER:
            models_state_val.classifier_state = ClassifierDeploy.parse_obj(json_data)
        elif models_state_val.current_model_type == ModelType.DETECTOR:
            models_state_val.detector_state = DetectorDeploy.parse_obj(json_data)
    except Exception:
        gr.Info("Error Parsing JSON", duration=2)

    return models_state_val


@append_flipped_bool_decorator
def update_class_detection_threshold(
    class_idx: int, models_state_val: DualModelInterface, threshold_val: float
) -> Tuple[DualModelInterface, int, bool]:
    models_state_val.detector_state.detector_configs[class_idx].detection_threshold = (
        threshold_val
    )
    return models_state_val, class_idx, True


@append_flipped_bool_decorator
def update_nms_threshold(
    models_state_val: DualModelInterface, nms_threshold_val: float
) -> DualModelInterface:
    models_state_val.detector_state.nms_threshold = nms_threshold_val
    return models_state_val


@append_flipped_bool_decorator
def update_class_label(
    class_idx: int, class_name: str, models_state_val: DualModelInterface
) -> DualModelInterface:
    current_configs = models_state_val.current_configs
    current_configs[class_idx].name = class_name
    if len(current_configs[class_idx].examples_to_include) == 0:
        current_configs[class_idx].examples_to_include = [class_name]

    return models_state_val


@append_flipped_bool_decorator
def update_include_example(
    class_idx: int,
    include_idx: int,
    label_name: str,
    models_state_val: DualModelInterface,
) -> DualModelInterface:
    current_configs = models_state_val.current_configs
    current_configs[class_idx].examples_to_include[include_idx] = label_name
    return models_state_val


@append_flipped_bool_decorator
def update_exclude_example(
    class_idx: int,
    exclude_idx: int,
    label_name: str,
    models_state_val: DualModelInterface,
) -> DualModelInterface:
    current_configs = models_state_val.current_configs
    current_configs[class_idx].examples_to_exclude[exclude_idx] = label_name
    return models_state_val


@append_flipped_bool_decorator
def change_example_count(
    class_idx: int,
    type_of_example: str,
    change_key: str,
    models_state_val: DualModelInterface,
) -> Tuple[DualModelInterface, int, bool]:
    # We expect `type_of_example` to be "to_include" or "to_exclude"
    # We expect `change_key` to be "increment" or "decrement"
    current_configs = models_state_val.current_configs

    if change_key == "increment":
        attr_to_modify = getattr(current_configs[class_idx], type_of_example)
        attr_to_modify.append(
            f"to include in class {class_idx}"
            if type_of_example == "examples_to_include"
            else f"to exclude in class {class_idx}"
        )
        setattr(current_configs[class_idx], type_of_example, attr_to_modify)

    elif change_key == "decrement":
        attr_to_modify = getattr(current_configs[class_idx], type_of_example)
        attr_to_modify = attr_to_modify[:-1]
        setattr(current_configs[class_idx], type_of_example, attr_to_modify)

    return models_state_val, class_idx, True


def classify_deploy_and_infer(
    to_display: Union[Image.Image, np.ndarray], models_state_val: DualModelInterface
) -> dict:
    if to_display is None:
        return {}
    try:
        deployed_id = deploy_classifier(models_state_val.classifier_state.dict())
        classify_results = get_classifier_results(to_display, deployed_id)
        return classify_results
    except json.decoder.JSONDecodeError as e:
        gr.Info(
            "JSON Decode Error in Model Deploy or Classification Response", duration=2
        )
        return {}
    except requests.exceptions.ConnectionError as ce:
        gr.Info(
            "Request Connection Error in Model Deploy or Classification Response",
            duration=2,
        )
        return {}
    except ValueError as ve:
        gr.Info("Malformatted model. Please update the configuration.", duration=2)
        return {}
    except Exception as e:
        gr.Info(
            "Generic Exception** in Model Deploy or Classification Response. Please try again.",
            duration=2,
        )
        return {}


def detect_deploy_and_infer(
    to_display: Image.Image, models_state_val: DualModelInterface
) -> list:
    if to_display is None:
        return [[]]
    try:
        deployed_id = deploy_detector(models_state_val.detector_state.dict())
        detection_results = get_detector_results(to_display, deployed_id)
        return detection_results
    except json.decoder.JSONDecodeError as e:
        gr.Info("JSON Decode Error in Model Deploy or Detection Response", duration=2)
        return [[]]
    except requests.exceptions.ConnectionError as ce:
        gr.Info(
            "Request Connection Error in Model Deploy or Detection Response",
            duration=2,
        )
        return [[]]
    except ValueError as ve:
        gr.Info("Malformatted model. Please update the configuration.", duration=2)
        return [[]]
    except Exception as e:
        gr.Info(
            "Generic Exception** in Model Deploy or Detection Response. Please try again.",
            duration=2,
        )
        return [[]]


def dual_model_infer(
    gradio_img: Union[Image.Image, np.ndarray, str],
    models_state_val: DualModelInterface,
) -> Tuple[Image.Image, gr.Label]:
    # deal with image typing
    if gradio_img is None:
        return gradio_img, gr.Label(visible=False)
    elif isinstance(gradio_img, np.ndarray):
        img_byte_arr = io.BytesIO()
        pil_image = Image.fromarray(gradio_img)
    elif isinstance(gradio_img, str):
        pil_image = Image.open(gradio_img)
    else:
        pil_image = gradio_img

    # deal with inference
    if models_state_val.current_model_type == ModelType.DETECTOR:
        detection_inference_results = detect_deploy_and_infer(
            to_display=pil_image, models_state_val=models_state_val
        )
        illustrated_image = display_bounding_boxes(
            pil_image=pil_image,
            dets=detection_inference_results[0],
            threshold=0,  # we already did threshold checks server-side
        )
        return illustrated_image, gr.Label(visible=False)
    elif models_state_val.current_model_type == ModelType.CLASSIFIER:
        classification_inference_results = classify_deploy_and_infer(
            pil_image, models_state_val
        )
        return pil_image, gr.Label(classification_inference_results, visible=True)
    else:
        gr.Info(
            "No valid model config. Please select Detector or Classifier from the Dropdown.",
            duration=5,
        )
        return pil_image, gr.Label(visible=False)


@append_flipped_bool_decorator
def update_models_state(
    dropdown_val: str, models_state_val: DualModelInterface
) -> DualModelInterface:
    if dropdown_val == "Detector":
        models_state_val.overwrite_detector()
    elif dropdown_val == "Classifier":
        models_state_val.overwrite_classifier()
    models_state_val.current_model_type = ModelType(dropdown_val)
    return models_state_val
