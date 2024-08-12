import os
from fastapi import (
    FastAPI, 
    Request,
    HTTPException,
    status
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import redis.asyncio as redis

from pydantic_models import (
    DeployResponse,
    HTTPExceptionResponse,
    ClassifierDeploy
)

app = FastAPI()

def grab_redis_endpoint(kwargs: dict = {}) -> str:
    host = kwargs.get("host", os.environ.get("CACHE_REDIS_HOST", "host.docker.internal"))
    username = kwargs.get("username", os.environ.get("CACHE_REDIS_USERNAME", "default"))
    password = kwargs.get("password", os.environ.get("CACHE_REDIS_PASSWORD", "default_password"))
    port = kwargs.get("port", os.environ.get("CACHE_REDIS_PORT", 6379))
    return f"redis://{username}:{password}@{host}:{port}"

@app.on_event("startup")
async def startup_event() -> None:        
    app.state.config_cache = await redis.from_url(f"{grab_redis_endpoint()}?decode_responses=True")
    print(f"Ping successful: {await app.state.config_cache.ping()}")

@app.on_event("shutdown")
async def shutdown_event() -> None:
    # TODO: Explicitly dump redis state to disk
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

@app.post("/deploy_detector")
def deploy_detector() -> dict:
    return {"message": "Hello, World!"}

@app.get("/classify")
def classify() -> dict:
    return {"message": "Hello, World!"}

@app.get("/detect")
def detect() -> dict:
    return {"message": "Hello, World!"}