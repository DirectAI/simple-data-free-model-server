import os
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
    Collection
)

from pydantic_models import (
    DeployResponse,
    HTTPExceptionResponse,
    ClassifierDeploy,
    ClassifierResponse,
    DetectorDeploy
)
from utils import raise_if_cannot_open

app = FastAPI()

def get_classifier_model_handle() -> None:
    pass

def generate_random_classifier_scores(labels: List[str]) -> Dict[str, Union[str, Dict[str, float]]]:
    to_return: Dict[str, Union[str, Dict[str, float]]] = {
        "scores": {},
        "raw_scores": {},
        "pred": ""
    }
    scores_dict: Dict[str, float] = {}
    raw_scores_dict: Dict[str, float] = {}
    for l in labels:
        scores_dict[l] = random.random()
        raw_scores_dict[l] = random.random()
    
    to_return["pred"] = random.choice(labels)
    to_return["scores"] = scores_dict
    to_return["raw_scores"] = raw_scores_dict
    return to_return

def grab_redis_endpoint(kwargs: dict = {}) -> str:
    host = kwargs.get("host", os.environ.get("CACHE_REDIS_HOST", "host.docker.internal"))
    username = kwargs.get("username", os.environ.get("CACHE_REDIS_USERNAME", "default"))
    password = kwargs.get("password", os.environ.get("CACHE_REDIS_PASSWORD", "default_password"))
    port = kwargs.get("port", os.environ.get("CACHE_REDIS_PORT", 6379))
    return f"redis://{username}:{password}@{host}:{port}"

async def grab_config(deployed_id: str) -> Dict[str, Union[str, Collection[str]]]:
    try:
        model_config = await app.state.config_cache.get(deployed_id)
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

@app.get("/detect")
def detect() -> dict:
    return {"message": "Hello, World!"}