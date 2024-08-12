from fastapi import FastAPI

app = FastAPI()

@app.post("/deploy_classifier")
def deploy_classifier():
    return {"message": "Hello, World!"}

@app.post("/deploy_detector")
def deploy_detector():
    return {"message": "Hello, World!"}

@app.get("/classify")
def classify():
    return {"message": "Hello, World!"}

@app.get("/detect")
def detect():
    return {"message": "Hello, World!"}