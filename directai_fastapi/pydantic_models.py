import uuid
import json
from logging_config import logger
from fastapi import HTTPException
from pydantic import BaseModel, conlist, Field
from typing import Optional, List, Dict, TypeVar, Type
import redis.asyncio as redis


class DeployResponse(BaseModel):
    deployed_id: str
    message: str


class HTTPExceptionResponse(BaseModel):
    status_code: int
    message: str
    data: Optional[str] = None


class SingleClassifierClass(BaseModel):
    name: str
    examples_to_include: List[str]
    examples_to_exclude: List[str] = []

    class Config:
        orm_mode = True


_ClassifierDeploy = TypeVar("_ClassifierDeploy", bound="ClassifierDeploy")


class ClassifierDeploy(BaseModel):
    classifier_configs: List[SingleClassifierClass]
    deployed_id: Optional[str] = None
    augment_examples: Optional[bool] = True

    class Config:
        orm_mode = True

    def build_config_dict(self) -> dict:
        logger.debug(f"Classifier Configs: {self.classifier_configs}")
        if len(self.classifier_configs) == 0:
            raise ValueError(
                f"Class list is empty.",
            )
        for classifier_config in self.classifier_configs:
            logger.debug(classifier_config.examples_to_include)
            if len(classifier_config.examples_to_include) == 0:
                raise ValueError(
                    f"Model lacks example_to_include for {classifier_config.name} class."
                )
        labels = [c.name for c in self.classifier_configs]
        inc_sub_labels_dict: dict[str, List[str]] = {
            c.name: c.examples_to_include for c in self.classifier_configs
        }
        exc_sub_labels_dict: dict[str, List[str]] = {
            c.name: c.examples_to_exclude for c in self.classifier_configs
        }
        config_dict = {
            "labels": labels,
            "inc_sub_labels_dict": inc_sub_labels_dict,
            "exc_sub_labels_dict": exc_sub_labels_dict,
            "augment_examples": self.augment_examples,
        }

        return config_dict

    @classmethod
    def from_config_dict(
        cls: Type[_ClassifierDeploy], config_dict: dict
    ) -> _ClassifierDeploy:
        labels = set(config_dict.get("labels", []))
        inc_keys = set(config_dict.get("inc_sub_labels_dict", {}).keys())
        exc_keys = set(config_dict.get("exc_sub_labels_dict", {}).keys())

        if labels != inc_keys or not exc_keys.issubset(labels):
            raise ValueError(
                "Labels and inc_sub_labels_dict keys must be equal. exc_sub_labels_dict keys must be a subset of labels."
            )

        classifier_configs = [
            SingleClassifierClass(
                name=k,
                examples_to_include=v,
                examples_to_exclude=(
                    config_dict["exc_sub_labels_dict"].get(k, [])
                    if config_dict.get("exc_sub_labels_dict")
                    else []
                ),
            )
            for k, v in config_dict["inc_sub_labels_dict"].items()
        ]
        return cls(
            classifier_configs=classifier_configs,
            augment_examples=config_dict.get("augment_examples", True),
            deployed_id=config_dict.get("deployed_id"),
        )

    async def save_configuration(self, config_cache: redis.Redis) -> dict:
        try:
            config_dict = self.build_config_dict()
        except ValueError as e:
            raise HTTPException(
                status_code=422,
                detail=f"{e}",
            )

        if self.deployed_id is not None:
            key_exists = await config_cache.exists(self.deployed_id)
        else:
            key_exists = False

        if not key_exists:
            if self.deployed_id is not None:
                message = f"{self.deployed_id} model not found. Generated new model."
            else:
                message = "New model deployed."
            self.deployed_id = str(uuid.uuid4())
        else:
            message = "Model updated."

        assert (
            self.deployed_id is not None
        ), "deployed_id should not be None at this point"
        config_dict["deployed_id"] = self.deployed_id
        await config_cache.set(self.deployed_id, json.dumps(config_dict))
        return {"deployed_id": self.deployed_id, "message": message}


class ClassifierResponse(BaseModel):
    scores: Dict[str, float]
    pred: str
    raw_scores: Dict[str, float]


class SingleDetectorClass(BaseModel):
    name: str
    examples_to_include: List[str]
    examples_to_exclude: List[str] = []
    detection_threshold: float = 0.1


class DetectorDeploy(BaseModel):
    detector_configs: List[SingleDetectorClass]
    nms_threshold: float = 0.4
    class_agnostic_nms: Optional[bool] = True
    deployed_id: Optional[str] = None
    augment_examples: Optional[bool] = True

    class Config:
        orm_mode = True

    @classmethod
    def from_config_dict(cls, config_dict: dict) -> "DetectorDeploy":
        labels = set(config_dict.get("labels", []))
        inc_keys = set(config_dict.get("inc_sub_labels_dict", {}).keys())
        exc_keys = set(config_dict.get("exc_sub_labels_dict", {}).keys())

        if labels != inc_keys or not exc_keys.issubset(labels):
            raise ValueError(
                "Labels and inc_sub_labels_dict keys must be equal. exc_sub_labels_dict keys must be a subset of labels."
            )

        detector_configs = [
            SingleDetectorClass(
                name=k,
                examples_to_include=v,
                examples_to_exclude=(
                    config_dict["exc_sub_labels_dict"].get(k, [])
                    if config_dict.get("exc_sub_labels_dict")
                    else []
                ),
                detection_threshold=config_dict["label_conf_thres"].get(k, 0.1),
            )
            for k, v in config_dict["inc_sub_labels_dict"].items()
        ]

        return cls(
            detector_configs=detector_configs,
            nms_threshold=config_dict.get("nms_threshold", 0.4),
            class_agnostic_nms=config_dict.get("class_agnostic_nms", True),
            augment_examples=config_dict.get("augment_examples", True),
            deployed_id=config_dict.get("deployed_id"),
        )

    async def save_configuration(self, config_cache: redis.Redis) -> dict:
        logger.info(f"Detector Configs: {self.detector_configs}")
        for detector_config in self.detector_configs:
            logger.info(detector_config.examples_to_include)
            if len(detector_config.examples_to_include) == 0:
                raise HTTPException(
                    status_code=422,
                    detail=f"Model lacks example_to_include for {detector_config.name} class.",
                )
        labels = [c.name for c in self.detector_configs]
        inc_sub_labels_dict: dict[str, List[str]] = {
            c.name: c.examples_to_include for c in self.detector_configs
        }
        exc_sub_labels_dict: dict[str, List[str]] = {
            c.name: c.examples_to_exclude for c in self.detector_configs
        }
        label_conf_thres: dict[str, float] = {
            c.name: c.detection_threshold for c in self.detector_configs
        }
        config_dict = {
            "labels": labels,
            "inc_sub_labels_dict": inc_sub_labels_dict,
            "exc_sub_labels_dict": exc_sub_labels_dict,
            "augment_examples": self.augment_examples,
            "nms_threshold": self.nms_threshold,
            "class_agnostic_nms": self.class_agnostic_nms,
            "label_conf_thres": label_conf_thres,
        }

        if self.deployed_id is not None:
            key_exists = await config_cache.exists(self.deployed_id)
        else:
            key_exists = False

        if not key_exists:
            if self.deployed_id is not None:
                message = f"{self.deployed_id} model not found. Generated new model."
            else:
                message = "New model deployed."
            self.deployed_id = str(uuid.uuid4())
        else:
            message = "Model updated."

        assert (
            self.deployed_id is not None
        ), "deployed_id should not be None at this point"
        config_dict["deployed_id"] = self.deployed_id
        await config_cache.set(self.deployed_id, json.dumps(config_dict))
        return {"deployed_id": self.deployed_id, "message": message}


class SingleDetectionResponse(BaseModel):
    # see discussion: https://github.com/pydantic/pydantic/issues/975
    tlbr: conlist(float, min_items=4, max_items=4)  # type: ignore[valid-type]
    score: float
    class_: str = Field(alias="class")

    class Config:
        allow_population_by_field_name = True
