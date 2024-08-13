import os
from fastapi import FastAPI, Request

from pydantic_models import (
    DeployResponse,
    HTTPExceptionResponse
)
from modeling.distributed_backend import (
    deploy_classifier_backend_model,
    deploy_detector_backend_model
)

app = FastAPI()

@app.on_event("startup")
async def startup_event() -> None:    
    app.state.classifier_handle = deploy_classifier_backend_model()
    app.state.detector_handle = deploy_detector_backend_model()

@app.on_event("shutdown")
async def shutdown_event() -> None:
    # TODO: Explicitly dump redis state to disk
    pass
    
@app.post("/deploy_classifier")
def deploy_classifier() -> dict:
    return {"message": "Hello, World!"}

@app.post("/deploy_detector")
def deploy_detector() -> dict:
    return {"message": "Hello, World!"}

@app.get("/classify")
async def classify() -> dict:
    label = await app.state.classifier_handle.remote(None)
    return {"label": label}

@app.get("/detect")
async def detect() -> dict:
    bboxes = await app.state.detector_handle.remote(None)
    return {"bboxes": bboxes}