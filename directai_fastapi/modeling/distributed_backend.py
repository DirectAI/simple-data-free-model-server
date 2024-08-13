from ray import serve
from ray.serve.handle import DeploymentHandle
from PIL import Image


serve.start(http_options={"port": 8100})


@serve.deployment
class ObjectDetector:
    async def __call__(self, image: Image.Image) -> list[list[float]]:
        # Placeholder implementation
        return [[0.0, 0.0, 1.0, 1.0]]


@serve.deployment
class ImageClassifier:
    async def __call__(self, image: Image.Image) -> str:
        # Placeholder implementation
        return "cat"


def deploy_classifier_backend_model() -> DeploymentHandle:
    classifier = ImageClassifier.bind()
    handle = serve.run(classifier, name="classifier", route_prefix=None)
    return handle


def deploy_detector_backend_model() -> DeploymentHandle:
    detector = ObjectDetector.bind()
    handle = serve.run(detector, name="detector", route_prefix=None)
    return handle