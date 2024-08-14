from ray import serve
from ray.serve import Deployment
from ray.serve.handle import DeploymentHandle
from PIL import Image

from pydantic_models import ClassifierResponse


serve.start(http_options={"port": 8100})


@serve.deployment
class ObjectDetector:
    async def __call__(self, image: Image.Image) -> list[list[float]]:
        # Placeholder implementation
        return [[0.0, 0.0, 1.0, 1.0]]


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