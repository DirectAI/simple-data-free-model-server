import random
from ray import serve
from ray.serve import Deployment
from ray.serve.handle import DeploymentHandle
from PIL import Image

from pydantic_models import ClassifierResponse, SingleDetectionResponse
from typing import List


serve.start(http_options={"port": 8100})


@serve.deployment
class ObjectDetector:
    async def __call__(self, image: Image.Image) -> List[List[SingleDetectionResponse]]:
        # Placeholder implementation
        sdr = SingleDetectionResponse(
            tlbr = [0.0, 0.0, 1.0, 1.0],
            score = random.random(),
            class_ = "dog"
        )
        return [[sdr]]

# def generate_random_detector_scores(labels: List[str]) -> List[List[SingleDetectionResponse]]:
#     to_return: List[SingleDetectionResponse] = []
#     for l in labels:
#         num_detections = random.randint(0, 5)
#         for d_idx in range(num_detections):
#             detection: Dict[str, Union[List[float], str, float]] = {
#                 "tlbr": [random.uniform(0, 1000) for _ in range(4)],
#                 "score": random.random(),
#                 "class": l
#             }
#             to_return.append(SingleDetectionResponse(**detection))
#     return [to_return]


@serve.deployment
class ImageClassifier:
    async def __call__(self, image: Image.Image) -> ClassifierResponse:
        # Placeholder implementation
        resp = ClassifierResponse(
            pred="dog",
            scores={"dog": 0.9, "cat": 0.1},
            raw_scores={"dog": 9, "cat": 0.5}
        )
        return resp


def deploy_classifier_backend_model() -> DeploymentHandle:
    assert isinstance(ImageClassifier, Deployment)
    classifier = ImageClassifier.bind()
    handle = serve.run(classifier, name="classifier", route_prefix=None)
    return handle


def deploy_detector_backend_model() -> DeploymentHandle:
    assert isinstance(ObjectDetector, Deployment)
    detector = ObjectDetector.bind()
    handle = serve.run(detector, name="detector", route_prefix=None)
    return handle