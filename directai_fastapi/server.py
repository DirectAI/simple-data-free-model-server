import os
import redis.asyncio as redis
from fastapi import FastAPI, Request

from pydantic_models import (
    DeployResponse,
    HTTPExceptionResponse,
    ClassifierDeploy,
    DetectorDeploy
)

app = FastAPI()

@app.on_event("startup")
async def startup_event() -> None:    
    pass

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
def classify() -> dict:
    return {"message": "Hello, World!"}

@app.get("/detect")
def detect() -> dict:
    return {"message": "Hello, World!"}