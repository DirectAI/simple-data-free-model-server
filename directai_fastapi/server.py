from fastapi import FastAPI

app = FastAPI()

@app.post("/deploy_classifier")
def deploy_classifier() -> dict:
    return {"message": "Hello, World!"}

@app.post("/deploy_detector")
def deploy_detector() -> dict:
    return {"message": "Hello, World!"}

# light change
@app.get("/classify")
def classify() -> dict:
    return {"message": "Hello, World!"}

@app.get("/detect")
def detect() -> dict:
    return {"message": "Hello, World!"}