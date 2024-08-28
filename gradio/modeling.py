import io
import requests
from PIL import Image
import numpy as np
from typing import Union, List, Optional
from pydantic import BaseModel

FASTAPI_HOST = "host.docker.internal"
FASTAPI_PORT = 8001
endpoint = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/"


class SingleClassifierClass(BaseModel):
    name: str
    examples_to_include: List[str]
    examples_to_exclude: List[str] = []

    class Config:
        from_attributes = True
        allow_population_by_field_name = True


class ClassifierDeploy(BaseModel):
    classifier_configs: List[SingleClassifierClass]
    deployed_id: Optional[str] = None
    augment_examples: Optional[bool] = True

    class Config:
        allow_population_by_field_name = True

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
        allow_population_by_field_name = True


class DetectorDeploy(BaseModel):
    detector_configs: List[SingleDetectorClass]
    nms_threshold: float = 0.4
    class_agnostic_nms: Optional[bool] = True
    deployed_id: Optional[str] = None
    augment_examples: Optional[bool] = True

    class Config:
        allow_population_by_field_name = True
        from_attributes = True

    def add_class(self, name: str = "") -> None:
        if name == "":
            num_classes = len(self.detector_configs)
            name = f"class_{num_classes}"
        new_detector_class = SingleDetectorClass(name=name, examples_to_include=[])
        self.detector_configs.append(new_detector_class)

    def remove_class(self) -> None:
        self.detector_configs = self.detector_configs[:-1]


class DualModelInterface:
    def __init__(self) -> None:
        self.current_model_type = ""
        self.classifier_state = ClassifierDeploy(classifier_configs=[])
        self.detector_state = DetectorDeploy(detector_configs=[])

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
        self.current_model_type = current_model_type

    def add_class(self, name: str = "") -> "DualModelInterface":
        if self.current_model_type == "Detector":
            self.detector_state.add_class(name=name)
        elif self.current_model_type == "Classifier":
            self.classifier_state.add_class(name=name)
        return self

    def remove_class(self) -> "DualModelInterface":
        if self.current_model_type == "Detector":
            self.detector_state.remove_class()
        elif self.current_model_type == "Classifier":
            self.classifier_state.remove_class()
        return self

    def __len__(self) -> int:
        if self.current_model_type == "Detector":
            return len(self.detector_state.detector_configs)
        elif self.current_model_type == "Classifier":
            return len(self.classifier_state.classifier_configs)
        return 0

    def _get_class_of_interest(
        self, idx: int
    ) -> Union[SingleClassifierClass, SingleDetectorClass]:
        assert self.current_model_type in [
            "Detector",
            "Classifier",
        ], "Invalid model type"
        if self.current_model_type == "Detector":
            return self.detector_state.detector_configs[idx]
        else:
            # i.e. self.current_model_type == "Classifier"
            # added else to provide coverage for mypy
            return self.classifier_state.classifier_configs[idx]

    def get_class_label(self, idx: int) -> str:
        class_of_interest = self._get_class_of_interest(idx)
        return class_of_interest.name

    def get_class_extoinc(self, idx: int) -> List[str]:
        class_of_interest = self._get_class_of_interest(idx)
        return class_of_interest.examples_to_include

    def get_class_extoexc(self, idx: int) -> List[str]:
        class_of_interest = self._get_class_of_interest(idx)
        return class_of_interest.examples_to_exclude

    def dict(self) -> dict:
        if self.current_model_type == "Detector":
            return self.detector_state.dict()
        else:
            # i.e. self.current_model_type == "Classifier"
            # added else to provide coverage for mypy
            return self.classifier_state.dict()


def deploy_classifier(classifier_config: dict) -> str:
    response = requests.post(endpoint + "deploy_classifier", json=classifier_config)
    if response.status_code == 500:
        raise ValueError(response.text)
    response_json = response.json()
    if response.status_code != 200:
        raise ValueError(response_json["message"])
    return response_json["deployed_id"]


def get_classifier_results(
    pil_image: Union[Image.Image, np.ndarray], deployed_id: str
) -> dict:
    if isinstance(pil_image, np.ndarray):
        img_byte_arr = io.BytesIO()
        pil_image = Image.fromarray(pil_image)
        pil_image.save(img_byte_arr, format="JPEG")
        img_byte_arr_val = img_byte_arr.getvalue()
    elif isinstance(pil_image, str):
        pil_image = Image.open(pil_image)
        img_byte_arr = io.BytesIO()
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
