from logging_config import logger
import os
import json
import random
from fastapi import FastAPI, Request, HTTPException, status, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from typing import Dict, List, Any, Union, Collection, Optional

from pydantic_models import (
    DeployResponse,
    HTTPExceptionResponse,
    ClassifierDeploy,
    ClassifierResponse,
    DetectorDeploy,
    SingleDetectionResponse,
)
from utils import raise_if_cannot_open
from modeling.distributed_backend import deploy_backend_models

app = FastAPI()


def grab_redis_endpoint(
    host: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    port: Optional[Union[int, str]] = None,
) -> str:
    if host is None:
        host = os.environ.get("CACHE_REDIS_HOST", "host.docker.internal")
    if username is None:
        username = os.environ.get("CACHE_REDIS_USERNAME", "default")
    if password is None:
        password = os.environ.get("CACHE_REDIS_PASSWORD", "default_password")
    if port is None:
        port = os.environ.get("CACHE_REDIS_PORT", 6379)
    return f"redis://{username}:{password}@{host}:{port}"


async def grab_config(deployed_id: str) -> Dict[str, Union[str, Collection[str]]]:
    try:
        model_config = await app.state.config_cache.get(deployed_id)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail="Exception in Model Config Storage Server. Please try again.",
        )
    if model_config is None:
        raise HTTPException(status_code=400, detail="Key not found")
    return json.loads(model_config)


@app.on_event("startup")
async def startup_event() -> None:
    app.state.detector_handle, app.state.classifier_handle = deploy_backend_models()
    app.state.config_cache = await redis.from_url(
        f"{grab_redis_endpoint()}?decode_responses=True"
    )
    logger.info(f"Ping successful: {await app.state.config_cache.ping()}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.config_cache.aclose()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    logger.info(f"{request}: {exc_str}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "message": exc_str,
            "data": None,
        },
    )


@app.exception_handler(HTTPException)
async def exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    exc_str = f"{exc.detail}".replace("\n", " ").replace("   ", " ")
    logger.info(f"{request}: {exc_str}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status_code": exc.status_code, "message": exc_str, "data": None},
    )


@app.post(
    "/deploy_classifier",
    include_in_schema=True,
    responses={
        200: {"model": DeployResponse},
        422: {"model": HTTPExceptionResponse},
        502: {"model": HTTPExceptionResponse},
    },
)
async def deploy_classifier(request: Request, config: ClassifierDeploy) -> dict:
    """
    Deploy New or Edit Existing Classifier with Natural Language. We expect at least one class definition.

    Optionally, provide the `deployed_id` of an existing classifier to modify its configuration in-place.
    """
    deploy_response = await config.save_configuration(
        config_cache=app.state.config_cache
    )
    logger.info(f"Deployed classifier w/ ID: {deploy_response['deployed_id']}")
    return deploy_response


@app.post(
    "/classify",
    include_in_schema=True,
    responses={
        200: {"model": ClassifierResponse},
        400: {"model": HTTPExceptionResponse},
        422: {"model": HTTPExceptionResponse},
        502: {"model": HTTPExceptionResponse},
    },
)
async def classify_examples(
    request: Request, deployed_id: str, data: UploadFile = File()
) -> Dict[str, Union[str, Dict[str, float]]]:
    """Get classification score from deployed model"""
    image = data.file.read()
    raise_if_cannot_open(image)
    logger.info(f"Got request for {deployed_id}, which is a classifier model")
    loaded_config = await grab_config(deployed_id)
    labels = loaded_config["labels"]
    assert isinstance(labels, list), "Labels should be a list of strings"
    inc_sub_labels_dict = loaded_config.get("inc_sub_labels_dict", None)
    exc_sub_labels_dict = loaded_config.get("exc_sub_labels_dict", None)
    augment_examples = loaded_config.get("augment_examples", True)

    scores = await app.state.classifier_handle.remote(
        image,
        labels=labels,
        inc_sub_labels_dict=inc_sub_labels_dict,
        exc_sub_labels_dict=exc_sub_labels_dict,
        augment_examples=augment_examples,
    )

    return scores


@app.post(
    "/deploy_detector",
    include_in_schema=True,
    responses={
        200: {"model": DeployResponse},
        422: {"model": HTTPExceptionResponse},
        502: {"model": HTTPExceptionResponse},
    },
)
async def deploy_detector(request: Request, config: DetectorDeploy) -> dict:
    """
    Deploy New or Edit Existing Detector with Natural Language. We expect at least one class definition.

    Optionally, provide the `deployed_id` of an existing object detector to modify its configuration in-place.
    """
    deploy_response = await config.save_configuration(
        config_cache=app.state.config_cache
    )
    logger.info(f"Deployed detector w/ ID: {deploy_response['deployed_id']}")
    return deploy_response


@app.post(
    "/detect",
    include_in_schema=True,
    responses={
        200: {"model": List[List[SingleDetectionResponse]]},
        400: {"model": HTTPExceptionResponse},
        422: {"model": HTTPExceptionResponse},
        502: {"model": HTTPExceptionResponse},
    },
)
async def run_detector(
    request: Request,
    deployed_id: str,
    data: UploadFile = File(),
) -> List[List[SingleDetectionResponse]]:
    """Get detections from deployed model"""
    image = data.file.read()
    raise_if_cannot_open(image)
    logger.info(f"Got request for {deployed_id}, which is a detector model")
    detector_configs = await grab_config(deployed_id)
    labels = detector_configs["labels"]
    assert isinstance(labels, list), "Labels should be a list of strings"
    inc_sub_labels_dict = detector_configs.get("inc_sub_labels_dict", None)
    exc_sub_labels_dict = detector_configs.get("exc_sub_labels_dict", None)
    label_conf_thres = detector_configs.get("label_conf_thres", None)
    augment_examples = detector_configs.get("augment_examples", True)
    nms_threshold = detector_configs.get("nms_threshold", 0.4)
    class_agnostic_nms = detector_configs.get("class_agnostic_nms", True)

    bboxes = await app.state.detector_handle.remote(
        image,
        labels=labels,
        inc_sub_labels_dict=inc_sub_labels_dict,
        exc_sub_labels_dict=exc_sub_labels_dict,
        label_conf_thres=label_conf_thres,
        augment_examples=augment_examples,
        nms_thre=nms_threshold,
        run_class_agnostic_nms=class_agnostic_nms,
    )

    return [
        bboxes,
    ]


@app.get(
    "/model",
    include_in_schema=True,
    responses={
        200: {"model": Union[DetectorDeploy, ClassifierDeploy]},
        400: {"model": HTTPExceptionResponse},
        422: {"model": HTTPExceptionResponse},
        502: {"model": HTTPExceptionResponse},
    },
)
async def get_model(
    request: Request, deployed_id: str
) -> Union[DetectorDeploy, ClassifierDeploy]:
    logger.info(f"Got request for {deployed_id}.")
    model_agnostic_config = await grab_config(deployed_id)
    is_detector = "nms_threshold" in model_agnostic_config
    if is_detector:
        deployed_detector = DetectorDeploy.from_config_dict(model_agnostic_config)
        deployed_detector.deployed_id = deployed_id
        return deployed_detector

    else:
        deployed_classifier = ClassifierDeploy.from_config_dict(model_agnostic_config)
        deployed_classifier.deployed_id = deployed_id
        return deployed_classifier
