import io
import requests
from PIL import Image
import numpy as np
from typing import Union

FASTAPI_HOST = "local_fastapi_ben"
endpoint = f"http://{FASTAPI_HOST}:8000/"


def deploy_classifier(set_of_classes: list) -> str:
    body = {"classifier_configs": set_of_classes}
    response = requests.post(endpoint + "deploy_classifier", json=body)
    response_json = response.json()
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
    response_json = response.json()
    return response_json["scores"]
