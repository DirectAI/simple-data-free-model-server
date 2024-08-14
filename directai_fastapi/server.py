import os
import time
import json
import random
from fastapi import (
    FastAPI, 
    Request,
    HTTPException,
    status,
    UploadFile,
    File
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import redis.asyncio as redis
from typing import (
    Dict, 
    List, 
    Any, 
    Union, 
    Collection,
    Optional
)

from pydantic_models import (
    DeployResponse,
    HTTPExceptionResponse,
    ClassifierDeploy,
    ClassifierResponse,
    DetectorDeploy,
    SingleDetectionResponse,
    VerboseDetectorConfig
)
from utils import (
    raise_if_cannot_open,
    get_classifier_model_handle,
    get_detector_model_handle,
    generate_random_classifier_scores,
    generate_random_detector_scores
)

app = FastAPI()

def grab_redis_endpoint(
    host: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    port: Optional[Union[int,str]] = None
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
        start_time = time.time()
        model_config = await app.state.config_cache.get(deployed_id)
        end_time = time.time()
        print("Config Get Latency:", end_time-start_time)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail = "Exception in Model Config Storage Server. Please try again."
        )
    if model_config is None:
        raise HTTPException(status_code=400, detail="Key not found")
    return json.loads(model_config)

@app.on_event("startup")
async def startup_event() -> None:        
    app.state.config_cache = await redis.from_url(f"{grab_redis_endpoint()}?decode_responses=True")
    print(f"Ping successful: {await app.state.config_cache.ping()}")

@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.config_cache.aclose()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    print(f"{request}: {exc_str}")
    return JSONResponse(
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
        content = {
            'status_code': status.HTTP_422_UNPROCESSABLE_ENTITY,
            'message': exc_str, 
            'data': None
        }
    )

@app.exception_handler(HTTPException)
async def exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    exc_str = f'{exc.detail}'.replace('\n', ' ').replace('   ', ' ')
    print(f"{request}: {exc_str}")
    return JSONResponse(
        status_code = exc.status_code,
        content = {
            'status_code': exc.status_code, 
            'message': exc_str, 
            'data': None
        }
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
        502: {"model": HTTPExceptionResponse}
    }
)
async def deploy_classifier(request: Request, config: ClassifierDeploy) -> dict:
    """
    Deploy New or Edit Existing Classifier with Natural Language. We expect at least one class definition.
    
    Optionally, provide the `deployed_id` of an existing classifier to modify its configuration in-place.
    """
    deploy_response = await config.save_configuration(config_cache = app.state.config_cache)
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
        502: {"model": HTTPExceptionResponse}
    }
)
async def classify_examples(
    request: Request,
    deployed_id: str, 
    data: UploadFile=File()
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
    # TODO: Build model hanlde
    get_classifier_model_handle()
    scores = generate_random_classifier_scores(labels)
    print(f"Got scores: {scores}")
    
    return scores


@app.post(
    "/deploy_detector", 
    include_in_schema=True, 
    responses={
        200: {"model": DeployResponse},
        401: {"model": HTTPExceptionResponse},
        403: {"model": HTTPExceptionResponse},
        422: {"model": HTTPExceptionResponse},
        429: {"model": HTTPExceptionResponse},
        502: {"model": HTTPExceptionResponse}
    }
)
async def deploy_detector(request: Request, config: DetectorDeploy) -> dict:
    """
    Deploy New or Edit Existing Detector with Natural Language. We expect at least one class definition.
    
    Optionally, provide the `deployed_id` of an existing object detector to modify its configuration in-place.
    """
    deploy_response = await config.save_configuration(config_cache = app.state.config_cache)
    print(f"Deployed detector w/ ID: {deploy_response['deployed_id']}")
    return deploy_response

@app.post(
    "/detect", 
    include_in_schema=True, 
    responses={
        200: {"model": List[List[SingleDetectionResponse]]},
        400: {"model": HTTPExceptionResponse},
        401: {"model": HTTPExceptionResponse},
        403: {"model": HTTPExceptionResponse},
        422: {"model": HTTPExceptionResponse},
        429: {"model": HTTPExceptionResponse},
        502: {"model": HTTPExceptionResponse}
    }
)
async def run_detector(
    request: Request,
    deployed_id: str,
    data: UploadFile=File(),
) -> List[List[SingleDetectionResponse]]:
    """Get detections from deployed model"""
    print(f"Got request for {deployed_id}, which is a detector model")
    image = data.file.read()
    raise_if_cannot_open(image)
    get_detector_model_handle()
    detector_configs = await grab_config(deployed_id)
    ## NOTE: This might break if we have embedded BaseModel-inheriting objects inside the json object
    verbose_detector_configs = [
        VerboseDetectorConfig(**json.loads(d) if isinstance(d, str) else d) 
        for d in detector_configs['detector_configs']
    ]
    print(f"augment_examples: {detector_configs.get('augment_examples', None)}")
    
    class_labels = [d.name for d in verbose_detector_configs]
    pred = generate_random_detector_scores(labels=class_labels)
    
    return pred