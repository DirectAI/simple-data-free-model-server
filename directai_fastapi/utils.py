from PIL import Image
import cv2
import random
import numpy as np
import io
from copy import deepcopy
from fastapi import HTTPException
from typing import (
    Dict, 
    List, 
    Union
)
from pydantic_models import SingleDetectionResponse

def validate_can_open_with_opencv(image: bytes) -> bool:
    try:
        image = deepcopy(image)
        _ = cv2.imdecode(np.frombuffer(image, np.uint8), -1)
        return True
    except Exception as e:
        print(f"Error trying to open image with OpenCV: {str(e)}")
        return False


def validate_can_open_with_pil(image: bytes) -> bool:
    try:
        image = deepcopy(image)
        _ = np.array(Image.open(io.BytesIO(image)))
        return True
    except Exception as e:
        print(f"Error trying to open image with PIL: {str(e)}")
        return False


def validate_can_open_with_any(image: bytes) -> bool:
    return validate_can_open_with_opencv(image) and validate_can_open_with_pil(image)


def raise_if_cannot_open(image: bytes) -> None:
    if not validate_can_open_with_any(image):
        # NOTE: should we be more verbose? we could in theory return the specific error message
        raise HTTPException(status_code=422, detail="Invalid image received, unable to open.")

def get_classifier_model_handle() -> None:
    pass

def get_detector_model_handle() -> None:
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

def generate_random_detector_scores(labels: List[str]) -> List[List[SingleDetectionResponse]]:
    to_return: List[SingleDetectionResponse] = []
    for l in labels:
        num_detections = random.randint(0, 5)
        for d_idx in range(num_detections):
            detection: Dict[str, Union[List[float], str, float]] = {
                "tlbr": [random.uniform(0, 1000) for _ in range(4)],
                "score": random.random(),
                "class": l
            }
            to_return.append(SingleDetectionResponse(detection))
    return [to_return]