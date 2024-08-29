import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Union, List, Tuple


def get_color(class_id: str) -> Tuple[int, int, int]:
    # Use a hash function to generate a unique and stable color for each class_id
    hash_object = hashlib.md5(class_id.encode())
    hex_dig = hash_object.hexdigest()
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    color = (b, g, r)  # OpenCV uses BGR format

    return color


def display_bounding_boxes(
    pil_image: Image.Image,
    dets: List[dict],
    threshold: float,
) -> Image.Image:
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()
    # image is assumed to be an OpenCV image
    colors = {}
    for bbox in dets:
        tlbr = bbox["tlbr"]
        score = bbox["score"]
        label = bbox["class"]
        top_left = (int(tlbr[0]), int(tlbr[1]))
        bottom_right = (int(tlbr[2]), int(tlbr[3]))

        if score >= threshold:
            if label not in colors:
                colors[label] = get_color(label)
            color = colors[label]
            # Set font scale and line thickness adaptively
            width, height = pil_image.size  # PIL image uses size attribute
            font_scale = max(min(height, width) / 1200, 0.3)
            thickness = int(max(min(height, width) / 300, 1))

            draw.rectangle((top_left, bottom_right), outline=color, width=thickness)
            draw.text(
                (top_left[0], top_left[1] - 10),
                f"{label} {score:.2f}",
                fill=color,
                font=font,
            )

    return pil_image
