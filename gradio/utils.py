import json
import requests

from typing import Tuple, Union, List, Optional
import numpy as np
from PIL import Image

from modeling import (
    SingleClassifierClass,
    SingleDetectorClass,
    ClassifierDeploy,
    DetectorDeploy,
    DualModelInterface,
    deploy_classifier,
    get_classifier_results,
)


def upload_file(
    file_bytes: bytes, models_state_val: DualModelInterface, proxy_bool: bool
) -> Tuple[DualModelInterface, bool]:
    json_data = json.loads(file_bytes.decode("utf-8"))

    if models_state_val.current_model_type == "Classifier":
        models_state_val.classifier_state = ClassifierDeploy.parse_obj(json_data)
    elif models_state_val.current_model_type == "Detector":
        models_state_val.detector_state = DetectorDeploy.parse_obj(json_data)

    return models_state_val, not proxy_bool


def update_class_detection_threshold(
    class_idx: int,
    models_state_val: DualModelInterface,
    threshold_val: float,
    proxy_bool: bool,
) -> Tuple[DualModelInterface, bool]:
    models_state_val.detector_state.detector_configs[class_idx].detection_threshold = (
        threshold_val
    )
    return models_state_val, not proxy_bool


def update_nms_threshold(
    models_state_val: DualModelInterface, nms_threshold_val: float, proxy_bool: bool
) -> Tuple[DualModelInterface, bool]:
    models_state_val.detector_state.nms_threshold = nms_threshold_val
    return models_state_val, not proxy_bool


def update_class_label(
    class_idx: int,
    class_name: str,
    models_state_val: DualModelInterface,
    proxy_bool: bool,
) -> Tuple[DualModelInterface, bool]:
    if models_state_val.current_model_type == "Classifier":
        models_state_val.classifier_state.classifier_configs[class_idx].name = (
            class_name
        )
        if (
            len(
                models_state_val.classifier_state.classifier_configs[
                    class_idx
                ].examples_to_include
            )
            == 0
        ):
            models_state_val.classifier_state.classifier_configs[
                class_idx
            ].examples_to_include = [class_name]

    elif models_state_val.current_model_type == "Detector":
        models_state_val.detector_state.detector_configs[class_idx].name = class_name
        if (
            len(
                models_state_val.detector_state.detector_configs[
                    class_idx
                ].examples_to_include
            )
            == 0
        ):
            models_state_val.detector_state.detector_configs[
                class_idx
            ].examples_to_include = [class_name]

    return models_state_val, not proxy_bool


def update_include_example(
    class_idx: int,
    include_idx: int,
    label_name: str,
    models_state_val: DualModelInterface,
    proxy_bool: bool,
) -> Tuple[DualModelInterface, bool]:
    if models_state_val.current_model_type == "Classifier":
        models_state_val.classifier_state.classifier_configs[
            class_idx
        ].examples_to_include[include_idx] = label_name
    elif models_state_val.current_model_type == "Detector":
        models_state_val.detector_state.detector_configs[class_idx].examples_to_include[
            include_idx
        ] = label_name
    return models_state_val, not proxy_bool


def update_exclude_example(
    class_idx: int,
    exclude_idx: int,
    label_name: str,
    models_state_val: DualModelInterface,
    proxy_bool: bool,
) -> Tuple[DualModelInterface, bool]:
    if models_state_val.current_model_type == "Classifier":
        models_state_val.classifier_state.classifier_configs[
            class_idx
        ].examples_to_exclude[exclude_idx] = label_name
    elif models_state_val.current_model_type == "Detector":
        models_state_val.detector_state.detector_configs[class_idx].examples_to_exclude[
            exclude_idx
        ] = label_name
    return models_state_val, not proxy_bool


def change_example_count(
    class_idx: int,
    type_of_example: str,
    change_key: str,
    models_state_val: DualModelInterface,
    proxy_bool: bool,
) -> Tuple[DualModelInterface, int, bool, bool]:
    # We expect `type_of_example` to be "to_include" or "to_exclude"
    # We expect `change_key` to be "increment" or "decrement"
    if models_state_val.current_model_type == "Classifier":
        if change_key == "increment":
            if type_of_example == "examples_to_include":
                attr_to_modify = getattr(
                    models_state_val.classifier_state.classifier_configs[class_idx],
                    type_of_example,
                )
                attr_to_modify.append(f"to include in class {class_idx}")
                setattr(
                    models_state_val.classifier_state.classifier_configs[class_idx],
                    type_of_example,
                    attr_to_modify,
                )

            elif type_of_example == "examples_to_exclude":
                attr_to_modify = getattr(
                    models_state_val.classifier_state.classifier_configs[class_idx],
                    type_of_example,
                )
                attr_to_modify.append(f"to exclude in class {class_idx}")
                setattr(
                    models_state_val.classifier_state.classifier_configs[class_idx],
                    type_of_example,
                    attr_to_modify,
                )

        elif change_key == "decrement":
            attr_to_modify = getattr(
                models_state_val.classifier_state.classifier_configs[class_idx],
                type_of_example,
            )
            attr_to_modify = attr_to_modify[:-1]
            setattr(
                models_state_val.classifier_state.classifier_configs[class_idx],
                type_of_example,
                attr_to_modify,
            )

    elif models_state_val.current_model_type == "Detector":
        if change_key == "increment":
            if type_of_example == "examples_to_include":
                attr_to_modify = getattr(
                    models_state_val.detector_state.detector_configs[class_idx],
                    type_of_example,
                )
                attr_to_modify.append(f"to include in class {class_idx}")
                setattr(
                    models_state_val.detector_state.detector_configs[class_idx],
                    type_of_example,
                    attr_to_modify,
                )

            elif type_of_example == "examples_to_exclude":
                attr_to_modify = getattr(
                    models_state_val.detector_state.detector_configs[class_idx],
                    type_of_example,
                )
                attr_to_modify.append(f"to exclude in class {class_idx}")
                setattr(
                    models_state_val.detector_state.detector_configs[class_idx],
                    type_of_example,
                    attr_to_modify,
                )

        elif change_key == "decrement":
            attr_to_modify = getattr(
                models_state_val.detector_state.detector_configs[class_idx],
                type_of_example,
            )
            attr_to_modify = attr_to_modify[:-1]
            setattr(
                models_state_val.detector_state.detector_configs[class_idx],
                type_of_example,
                attr_to_modify,
            )

    return models_state_val, class_idx, True, not proxy_bool


def deploy_and_infer(
    to_display: Union[Image.Image, np.ndarray], models_state_val: DualModelInterface
) -> Tuple[dict, str]:
    if to_display is None:
        return {}, ""
    try:
        deployed_id = deploy_classifier(models_state_val.classifier_state.dict())
        classify_results = get_classifier_results(to_display, deployed_id)
        return classify_results, ""
    except json.decoder.JSONDecodeError as e:
        return {}, "**JSON Decode Error** in Model Deploy or Classification Response"
    except requests.exceptions.ConnectionError as ce:
        return (
            {},
            "**Request Connection Error** in Model Deploy or Classification Response",
        )
    except ValueError as ve:
        return {}, str(ve)
    except Exception as e:
        return (
            {},
            "**Generic Exception** in Model Deploy or Classification Response. Please try again.",
        )


def update_models_state(
    dropdown_val: str, models_state_val: DualModelInterface, proxy_bool: bool
) -> Tuple[DualModelInterface, bool]:
    if dropdown_val == "Detector":
        models_state_val.overwrite_detector()
    elif dropdown_val == "Classifier":
        models_state_val.overwrite_classifier()
    models_state_val.current_model_type = dropdown_val
    return models_state_val, not proxy_bool
