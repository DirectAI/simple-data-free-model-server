import random
from ray import serve
from ray.serve import Deployment
from ray.serve.handle import DeploymentHandle
from PIL import Image
import torch
from torch.nn import functional as F

from typing import List
from pydantic_models import ClassifierResponse, SingleDetectionResponse
from modeling.image_classifier import ZeroShotImageClassifierWithFeedback


serve.start(http_options={"port": 8100})


@serve.deployment
class ObjectDetector:
    async def __call__(self, image: Image.Image) -> List[List[SingleDetectionResponse]]:
        # Placeholder implementation
        single_detection = {
            "tlbr": [0.0, 0.0, 1.0, 1.0],
            "score": random.random(),
            "class": "dog",
        }
        sdr = SingleDetectionResponse.parse_obj(single_detection)
        return [[sdr]]


@serve.deployment
class ImageClassifier:
    def __init__(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ZeroShotImageClassifierWithFeedback(device=device)

    async def __call__(
        self,
        image: bytes,
        labels: list[str],
        inc_sub_labels_dict: dict[str, list[str]],
        exc_sub_labels_dict: dict[str, list[str]] | None = None,
        augment_examples: bool = True,
    ) -> ClassifierResponse:

        with torch.no_grad(), torch.autocast(str(self.model.device)):
            raw_scores = self.model(
                image,
                labels=labels,
                inc_sub_labels_dict=inc_sub_labels_dict,
                exc_sub_labels_dict=exc_sub_labels_dict,
                augment_examples=augment_examples,
            )

            # since we're only accepting the bytes version of image, we can assume the output has batch size 1
            raw_scores = raw_scores.squeeze(0)
            scores = F.softmax(raw_scores / 0.07, dim=0)
            ind = scores.argmax().item()
            assert isinstance(ind, int)  # this is just to make mypy happy
            pred = labels[ind]

        resp = ClassifierResponse(
            scores={label: score.item() for label, score in zip(labels, scores)},
            pred=pred,
            raw_scores={
                label: score.item() for label, score in zip(labels, raw_scores)
            },
        )

        return resp


assert isinstance(ObjectDetector, Deployment)
assert isinstance(ImageClassifier, Deployment)
MODELS_TO_DEPLOY: list[Deployment] = [ObjectDetector, ImageClassifier]


def get_deployment_name(deployment: Deployment) -> str:
    func_or_class = deployment.func_or_class
    if isinstance(func_or_class, str):
        return func_or_class
    else:
        return func_or_class.__name__


def deploy_backend_models(
    models_to_deploy: list[Deployment] | None = None,
) -> list[DeploymentHandle]:
    if models_to_deploy is None:
        models_to_deploy = MODELS_TO_DEPLOY

    handles = []

    for model in models_to_deploy:
        assert isinstance(model, Deployment)

    # if there is no GPU available, deploy one copy of each model on the CPU
    if not torch.cuda.is_available():
        for model in models_to_deploy:
            model_instance = model.bind()
            handle = serve.run(
                model_instance, name=get_deployment_name(model), route_prefix=None
            )
            handles.append(handle)
    else:
        # we would like to deploy models such that there is one copy of each on each GPU
        # ray doesn't have a nice way of doing this automatically
        # so we do a hacky workaround here
        # we have a list of models and we deploy them one by one
        # with resource requirements that ensure each model has exactly one replica on each GPU
        num_gpus = torch.cuda.device_count()

        per_gpu_remaining_resource = 1.0
        resource_usage_ratio = (
            0.75  # each model will be assigned 75% of the remaining resource
        )

        for model in models_to_deploy:
            resource_allocation = per_gpu_remaining_resource * resource_usage_ratio
            model = model.options(
                num_replicas=num_gpus,
                ray_actor_options={"num_gpus": resource_allocation},
            )
            model_instance = model.bind()
            handle = serve.run(
                model_instance, name=get_deployment_name(model), route_prefix=None
            )
            handles.append(handle)
            per_gpu_remaining_resource -= resource_allocation

    return handles
