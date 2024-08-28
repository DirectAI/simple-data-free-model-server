import io
import requests
from PIL import Image
import numpy as np
from typing import Union

FASTAPI_HOST = "host.docker.internal"
FASTAPI_PORT = 8001
endpoint = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/"


def deploy_classifier(classifier_config: dict) -> str:
    response = requests.post(endpoint + "deploy_classifier", json=classifier_config)
    if response.status_code == 500:
        raise ValueError(response.text)
    response_json = response.json()
    if response.status_code != 200:
        raise ValueError(response_json["message"])
    return response_json["deployed_id"]


def get_classifier_results(
    pil_image: Union[Image.Image, np.ndarray], deployed_id: str
) -> dict:
    if isinstance(pil_image, np.ndarray):
        img_byte_arr = io.BytesIO()
        pil_image = Image.fromarray(pil_image)
        pil_image.save(img_byte_arr, format="JPEG")
        img_byte_arr_val = img_byte_arr.getvalue()
    elif isinstance(pil_image, str):
        pil_image = Image.open(pil_image)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format="JPEG")
        img_byte_arr_val = img_byte_arr.getvalue()

    files = {
        "data": ("image.jpg", img_byte_arr_val, "image/jpeg"),
    }

    params = {"deployed_id": deployed_id}

    response = requests.post(endpoint + "classify", params=params, files=files)
    if response.status_code == 500:
        raise ValueError(response.text)
    response_json = response.json()
    if response.status_code != 200:
        raise ValueError(response_json["message"])
    return response_json["scores"]
