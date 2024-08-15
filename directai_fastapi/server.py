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
)
from utils import (
    raise_if_cannot_open,
)
from modeling.distributed_backend import (
    deploy_classifier_backend_model,
    deploy_detector_backend_model,
)

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
    app.state.classifier_handle = deploy_classifier_backend_model()
    app.state.detector_handle = deploy_detector_backend_model()
    app.state.config_cache = await redis.from_url(
        f"{grab_redis_endpoint()}?decode_responses=True"
    )
    print(f"Ping successful: {await app.state.config_cache.ping()}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.config_cache.aclose()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    print(f"{request}: {exc_str}")
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
    print(f"{request}: {exc_str}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"status_code": exc.status_code, "message": exc_str, "data": None},
    )


@app.post(
    "/deploy_classifier",
    include_in_schema=True,
    responses={
        200: {"model": DeployResponse},
        401: {"model": HTTPExceptionResponse},
        403: {"model": HTTPExceptionResponse},
        422: {"model": HTTPExceptionResponse},
        429: {"model": HTTPExceptionResponse},
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
    print(f"Deployed classifier w/ ID: {deploy_response['deployed_id']}")
    return deploy_response


@app.post(
    "/classify",
    include_in_schema=True,
    responses={
        200: {"model": ClassifierResponse},
        400: {"model": HTTPExceptionResponse},
        401: {"model": HTTPExceptionResponse},
        403: {"model": HTTPExceptionResponse},
        422: {"model": HTTPExceptionResponse},
        429: {"model": HTTPExceptionResponse},
        502: {"model": HTTPExceptionResponse},
    },
)
async def classify_examples(
    request: Request, deployed_id: str, data: UploadFile = File()
) -> Dict[str, Union[str, Dict[str, float]]]:
    """Get classification score from deployed model"""
    image = data.file.read()
    raise_if_cannot_open(image)
    print(f"Got request for {deployed_id}, which is a classifier model")
    loaded_config = await grab_config(deployed_id)
    labels = loaded_config["labels"]
    assert isinstance(labels, list), "Labels should be a list of strings"
    inc_sub_labels_dict = loaded_config.get("inc_sub_labels_dict", None)
    exc_sub_labels_dict = loaded_config.get("exc_sub_labels_dict", None)
    controls = loaded_config.get("controls", None)
    augment_examples = loaded_config.get("augment_examples", True)

    # TODO: run actual classifier model
    scores = await app.state.classifier_handle.remote(None)

    return scores


@app.get("/deploy_detector")
def deploy_detector() -> dict:
    return {"message": "Hello, World!"}


@app.get("/detect")
async def detect() -> dict:
    bboxes = await app.state.detector_handle.remote(None)
    return {"bboxes": bboxes}
