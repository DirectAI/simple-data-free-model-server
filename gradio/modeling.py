import io
import requests
from PIL import Image
import numpy as np
from enum import Enum
from typing import Union, List, Optional, Any, Callable
from pydantic import BaseModel

FASTAPI_HOST = "host.docker.internal"
FASTAPI_PORT = 8000
endpoint = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/"


# See notes on `change_this_bool_to_force_reload` in gradio/interface.py
# when gradio pushes a fix we can remove all decorators and maintain functionality
def append_flipped_bool_decorator(func: Callable) -> Callable:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "proxy_bool" in kwargs:
            proxy_bool = kwargs.pop("proxy_bool")
        else:
            proxy_bool = args[-1]
            args = args[:-1]
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return result + (not proxy_bool,)
        return result, not proxy_bool

    return wrapper


class SingleClassifierClass(BaseModel):
    name: str
    examples_to_include: List[str]
    examples_to_exclude: List[str] = []

    class Config:
        from_attributes = True
        populate_by_name = True


class ClassifierDeploy(BaseModel):
    classifier_configs: List[SingleClassifierClass]
    deployed_id: Optional[str] = None
    augment_examples: Optional[bool] = True

    class Config:
        populate_by_name = True

    def add_class(self, name: str = "") -> None:
        if name == "":
            num_classes = len(self.classifier_configs)
            name = f"class_{num_classes}"
        new_classifier_class = SingleClassifierClass(name=name, examples_to_include=[])
        self.classifier_configs.append(new_classifier_class)

    def remove_class(self) -> None:
        self.classifier_configs = self.classifier_configs[:-1]


class SingleDetectorClass(BaseModel):
    name: str
    examples_to_include: List[str]
    examples_to_exclude: List[str] = []
    detection_threshold: float = 0.1

    class Config:
        populate_by_name = True


class DetectorDeploy(BaseModel):
    detector_configs: List[SingleDetectorClass]
    nms_threshold: float = 0.4
    class_agnostic_nms: Optional[bool] = True
    deployed_id: Optional[str] = None
    augment_examples: Optional[bool] = True

    class Config:
        populate_by_name = True
        from_attributes = True

    def add_class(self, name: str = "") -> None:
        if name == "":
            num_classes = len(self.detector_configs)
            name = f"class_{num_classes}"
        new_detector_class = SingleDetectorClass(name=name, examples_to_include=[])
        self.detector_configs.append(new_detector_class)

    def remove_class(self) -> None:
        self.detector_configs = self.detector_configs[:-1]


class ModelType(Enum):
    DETECTOR = "Detector"
    CLASSIFIER = "Classifier"


class DualModelInterface:
    def __init__(self) -> None:
        self.current_model_type: Optional[ModelType] = None
        self.classifier_state = ClassifierDeploy(classifier_configs=[])
        self.detector_state = DetectorDeploy(detector_configs=[])

    @property
    def current_configs(
        self,
    ) -> Union[List[SingleClassifierClass], List[SingleDetectorClass]]:
        if self.current_model_type == ModelType.DETECTOR:
            return self.detector_state.detector_configs
        elif self.current_model_type == ModelType.CLASSIFIER:
            return self.classifier_state.classifier_configs
        else:
            raise ValueError("Model type is undefined. Can't obtain current configs.")

    @property
    def current_state(self) -> Union[ClassifierDeploy, DetectorDeploy]:
        if self.current_model_type == ModelType.DETECTOR:
            return self.detector_state
        elif self.current_model_type == ModelType.CLASSIFIER:
            return self.classifier_state
        else:
            raise ValueError("Model type is undefined. Can't obtain current state.")

    def overwrite_classifier(self) -> None:
        self.classifier_state.classifier_configs = [
            SingleClassifierClass(
                name=d.name,
                examples_to_include=d.examples_to_include,
                examples_to_exclude=d.examples_to_exclude,
            )
            for d in self.detector_state.detector_configs
        ]

    def overwrite_detector(self) -> None:
        self.detector_state.detector_configs = [
            SingleDetectorClass(
                name=c.name,
                examples_to_include=c.examples_to_include,
                examples_to_exclude=c.examples_to_exclude,
            )
            for c in self.classifier_state.classifier_configs
        ]

    def set_current_model_type(self, current_model_type: str) -> None:
        self.current_model_type = ModelType(current_model_type)

    @append_flipped_bool_decorator
    def add_class(self, name: str = "") -> "DualModelInterface":
        self.current_state.add_class(name=name)
        return self

    @append_flipped_bool_decorator
    def remove_class(self) -> "DualModelInterface":
        self.current_state.remove_class()
        return self

    def __len__(self) -> int:
        if self.current_model_type == ModelType.DETECTOR:
            return len(self.detector_state.detector_configs)
        elif self.current_model_type == ModelType.CLASSIFIER:
            return len(self.classifier_state.classifier_configs)
        else:
            # model type is unassigned at __init__
            return 0

    def _get_class_of_interest(
        self, idx: int
    ) -> Union[SingleClassifierClass, SingleDetectorClass]:
        if (
            self.current_model_type == ModelType.DETECTOR
            or self.current_model_type == ModelType.CLASSIFIER
        ):
            return self.current_configs[idx]
        else:
            raise ValueError("Model type is undefined. Can't obtain class of interest.")

    def get_class_label(self, idx: int) -> str:
        class_of_interest = self._get_class_of_interest(idx)
        return class_of_interest.name

    def get_class_extoinc(self, idx: int) -> List[str]:
        class_of_interest = self._get_class_of_interest(idx)
        return class_of_interest.examples_to_include

    def get_class_extoexc(self, idx: int) -> List[str]:
        class_of_interest = self._get_class_of_interest(idx)
        return class_of_interest.examples_to_exclude

    def full_model_dict(self) -> dict:
        if (
            self.current_model_type == ModelType.DETECTOR
            or self.current_model_type == ModelType.CLASSIFIER
        ):
            return self.current_state.dict()
        else:
            raise ValueError("Model type is undefined. Can't obtain model dict.")

    def display_dict(self) -> dict:
        display_dict = self.full_model_dict()
        if self.current_model_type == ModelType.CLASSIFIER:
            keys_to_delete = ["deployed_id", "augment_examples"]
        elif self.current_model_type == ModelType.DETECTOR:
            keys_to_delete = ["class_agnostic_nms", "deployed_id", "augment_examples"]
        else:
            raise ValueError("Model type is undefined. Can't obtain visible dict.")

        for key in keys_to_delete:
            if key in display_dict:
                del display_dict[key]
        return display_dict


def deploy_classifier(classifier_config: dict) -> str:
    response = requests.post(endpoint + "deploy_classifier", json=classifier_config)
    response_json = response.json()
    if response.status_code == 422:
        raise ValueError(response_json["message"])
    elif response.status_code == 500:
        raise RuntimeError(response.text)
    elif response.status_code != 200:
        raise Exception(response_json["message"])
    return response_json["deployed_id"]


def get_classifier_results(
    pil_image: Union[Image.Image, np.ndarray, str], deployed_id: str
) -> dict:
    img_byte_arr = io.BytesIO()
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)
    elif isinstance(pil_image, str):
        pil_image = Image.open(pil_image)
    pil_image.save(img_byte_arr, format="JPEG")
    img_byte_arr_val = img_byte_arr.getvalue()

    files = {
        "data": ("image.jpg", img_byte_arr_val, "image/jpeg"),
    }

    params = {"deployed_id": deployed_id}

    response = requests.post(endpoint + "classify", params=params, files=files)
    if response.status_code == 500:
        raise ValueError(response.text)
    response_json = response.json()
    if response.status_code != 200:
        raise ValueError(response_json["message"])
    return response_json["scores"]


def deploy_detector(detector_config: dict) -> str:
    response = requests.post(endpoint + "deploy_detector", json=detector_config)
    response_json = response.json()
    if response.status_code == 422:
        raise ValueError(response_json["message"])
    elif response.status_code == 500:
        raise RuntimeError(response.text)
    elif response.status_code != 200:
        raise Exception(response_json["message"])
    return response_json["deployed_id"]


def get_detector_results(
    pil_image: Union[Image.Image, np.ndarray, str], deployed_id: str
) -> list:
    img_byte_arr = io.BytesIO()
    if isinstance(pil_image, np.ndarray):
        pil_image = Image.fromarray(pil_image)
    elif isinstance(pil_image, str):
        pil_image = Image.open(pil_image)
    pil_image.save(img_byte_arr, format="JPEG")
    img_byte_arr_val = img_byte_arr.getvalue()

    files = {
        "data": ("image.jpg", img_byte_arr_val, "image/jpeg"),
    }

    params = {"deployed_id": deployed_id}

    response = requests.post(endpoint + "detect", params=params, files=files)
    if response.status_code == 500:
        raise ValueError(response.text)
    response_json = response.json()
    if response.status_code != 200:
        raise ValueError(response_json["message"])
    return response_json
