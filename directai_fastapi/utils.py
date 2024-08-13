from PIL import Image
import cv2
import numpy as np
import io
from copy import deepcopy
from fastapi import HTTPException


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