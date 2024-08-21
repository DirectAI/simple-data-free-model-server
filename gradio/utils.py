import json
from typing import Tuple, Union
import numpy as np
from PIL import Image

from modeling import deploy_classifier, get_classifier_results


def upload_file(file_bytes: bytes) -> Union[dict, list]:
    json_data = json.loads(file_bytes.decode("utf-8"))
    return json_data


def update_class_label(
    class_idx: int,
    class_name: str,
    class_states_val: list,
) -> list:
    class_states_val[class_idx]["name"] = class_name
    if len(class_states_val[class_idx]["examples_to_include"]) == 0:
        class_states_val[class_idx]["examples_to_include"] = [class_name]
    return class_states_val


def update_include_example(
    class_idx: int,
    include_idx: int,
    label_name: str,
    class_states_val: list,
) -> None:
    class_states_val[class_idx]["examples_to_include"][include_idx] = label_name


def update_exclude_example(
    class_idx: int,
    exclude_idx: int,
    label_name: str,
    class_states_val: list,
) -> None:
    class_states_val[class_idx]["examples_to_exclude"][exclude_idx] = label_name


def change_example_count(
    class_idx: int,
    type_of_example: str,
    change_key: str,
    class_states_val: list,
) -> Tuple[list, int, bool]:
    # We expect `type_of_example` to be "to_include" or "to_exclude"
    # We expect `change_key` to be "increment" or "decrement"
    if change_key == "increment":
        if type_of_example == "examples_to_include":
            class_states_val[class_idx][type_of_example].append(
                f"to include in class {class_idx}"
            )
        elif type_of_example == "examples_to_exclude":
            class_states_val[class_idx][type_of_example].append(
                f"to exclude from class {class_idx}"
            )

    elif change_key == "decrement":
        class_states_val[class_idx][type_of_example] = class_states_val[class_idx][
            type_of_example
        ][:-1]
    return class_states_val, class_idx, True


def deploy_and_infer(
    to_display: Union[Image.Image, np.ndarray], set_of_classes: list
) -> dict:
    if to_display is None:
        return {}
    deployed_id = deploy_classifier(set_of_classes)
    classify_results = get_classifier_results(to_display, deployed_id)
    return classify_results
