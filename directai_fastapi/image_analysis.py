import os
import base64
import requests
from openai import OpenAI
from PIL import Image

api_key = "sk-OMp69M01nQCIeldbG73BZaMhOlXcellpT7jEKp0F1ET3BlbkFJGtnUwAvoSmh1b9PFfsk4nPJT5iakC8nPTIXkgCpl8A"
client = OpenAI(api_key=api_key)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(image_path, class_a, class_b):
    base64_image = encode_image(image_path)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    true_class = class_a if class_a in image_path else class_b
    false_class = class_b if class_a == true_class else class_a
    prompt_text = f"""
    You are an expert in image recognition and computer vision. Explain why this is a {true_class} and not a {false_class}. Focus on aspects like color, texture, ingredients, presentation, and any other distinguishing characteristics. Do so in two sentences or fewer.
    """
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    response_json = response.json()

    analysis = response_json["choices"][0]["message"]["content"]
    return analysis


directory = "to_review"
class_a = "steak"
class_b = "filet_mignon"
analyses = []

import random

image_files = [
    f
    for f in os.listdir(directory)
    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
]
random.shuffle(image_files)
selected_files = image_files[:3]

from pydantic import BaseModel, conlist


class ClassConfig(BaseModel):
    name: str
    examples_to_include: list[str]
    examples_to_exclude: list[str]


class ClassifierConfig(BaseModel):
    classifier_configs: list[ClassConfig]


prompt_to_use = """
I'm using a computer vision tool that defines a classifier using a JSON as described below. I use a list of string prompts in `examples_to_include` and `examples_to_exclude` to define a class. Note that these examples should sufficiently describe the class, not components of the class. For example, a "foot" doesn't describe "leg" but "hairy leg" would. It would only be useful to add "foot" to `examples_to_include` if I had a number of false positives of feet being described as legs.

```json
{
    "classifier_configs": [
        {
            "name": "grilled_salmon",
            "examples_to_include": [
                "grilled salmon"
            ],
            "examples_to_exclude": []
        },
        {
            "name": "hot_and_sour_soup",
            "examples_to_include": [
                "hot and sour soup"
            ],
            "examples_to_exclude": []
        },
]
}
```

Given the following analysis on images that the model mis-classified, please propose an update for these two classes: steak and filet mignon. You may assume that the class name, with the exception of the underscores, is in `examples_to_include`. Please don't use more than three examples in any given include/exclude list.
"""


def summarize_and_recommend(analyses, client=client):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": prompt_to_use},
            {"role": "user", "content": f"{analyses}"},
        ],
        response_format=ClassifierConfig,
    )

    return completion.choices[0].message.parsed


analyses = []
for filename in selected_files:
    image_path = os.path.join(directory, filename)
    analysis = analyze_image(image_path=image_path, class_a=class_a, class_b=class_b)
    analyses.append({"image": filename, "analysis": analysis})

summary_and_recommendations = summarize_and_recommend(analyses)
print(summary_and_recommendations.json())
