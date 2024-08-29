import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_max  # type: ignore
import open_clip  # type: ignore
from functools import partial

from modeling.tensor_utils import (
    batch_encode_cache_missed_list_elements,
    image_bytes_to_tensor,
    squish_labels,
)
from modeling.prompt_templates import noop_hypothesis_formats, many_hypothesis_formats
from lru import LRU


class ZeroShotImageClassifierWithFeedback(nn.Module):
    def __init__(
        self,
        base_model_name: str = "ViT-H-14-quickgelu",
        dataset_name: str = "dfn5b",
        max_text_batch_size: int = 512,
        max_image_batch_size: int = 512,
        device: torch.device | str = "cuda",
        lru_cache_size: int = 4096,  # set to 0 to disable caching
        jit: bool = True,
        fp16: bool = True,
    ):
        super().__init__()

        self.device = torch.device(device) if type(device) is str else device
        self.fp16 = fp16

        # TODO: just do create_model, not create_model_and_transforms
        self.model, _, _ = open_clip.create_model_and_transforms(
            base_model_name,
            pretrained=dataset_name,
            jit=jit,
            image_resize_mode="squash",
            precision="fp16" if fp16 else "fp32",
        )
        self.tokenizer = open_clip.get_tokenizer(base_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.max_text_batch_size = max_text_batch_size
        self.max_image_batch_size = max_image_batch_size

        # we cache the text embeddings to avoid recomputing them
        # we use an LRU cache to avoid running out of memory
        # especially because likely the tensors will be large and stored in GPU memory
        self.augmented_label_encoding_cache: LRU | None = (
            LRU(lru_cache_size) if lru_cache_size > 0 else None
        )
        self.not_augmented_label_encoding_cache: LRU | None = (
            LRU(lru_cache_size) if lru_cache_size > 0 else None
        )

        preprocess_config = self.model.visual.preprocess_cfg
        self.img_mean = (
            torch.tensor(preprocess_config["mean"]).view(1, 3, 1, 1).to(self.device)
        )
        self.img_std = (
            torch.tensor(preprocess_config["std"]).view(1, 3, 1, 1).to(self.device)
        )
        self.img_size = preprocess_config["size"]
        if type(self.img_size) is int:
            self.img_size = (self.img_size, self.img_size)

    def encode_image(self, image: torch.Tensor | bytes) -> torch.Tensor:
        # enable to work with raw file instead of PIL image to save bandwidth during remote gRPC call
        if isinstance(image, bytes):
            image = image_bytes_to_tensor(image, self.img_size)

        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # NOTE: we are doing the normalization here instead of the data loader
        # to take advantage of easier access to the model's specific normalization values
        image = image.float() / 255.0
        image -= self.img_mean
        image /= self.img_std

        if self.fp16:
            image = image.half()

        feature_list = []
        for i in range(0, image.size(0), self.max_image_batch_size):
            features_subset = self.model.encode_image(
                image[i : i + self.max_image_batch_size]
            )
            feature_list.append(features_subset)

        features = torch.cat(feature_list, dim=0)
        features /= torch.norm(features, dim=1, keepdim=True)

        return features

    def _encode_text(self, text: list[str], augment: bool = True) -> torch.Tensor:
        # we apply the prompt templates commonly used with CLIP-based models unless otherwise specified
        templates = many_hypothesis_formats if augment else noop_hypothesis_formats
        augmented_text = [template.format(t) for t in text for template in templates]

        tokenized = self.tokenizer(augmented_text).to(self.device)

        features_list = []
        for i in range(0, len(tokenized), self.max_text_batch_size):
            features_subset = self.model.encode_text(
                tokenized[i : i + self.max_text_batch_size]
            )
            features_list.append(features_subset)

        features = torch.cat(features_list, dim=0)
        features /= torch.norm(features, dim=1, keepdim=True)
        features = features.view(len(text), len(templates), features.shape[1])
        features = features.mean(dim=1)
        features /= torch.norm(features, dim=1, keepdim=True)

        return features

    def encode_text(self, text: list[str], augment: bool = True) -> torch.Tensor:
        if augment:
            return batch_encode_cache_missed_list_elements(
                partial(self._encode_text, augment=True),
                text,
                self.augmented_label_encoding_cache,
            )
        else:
            return batch_encode_cache_missed_list_elements(
                partial(self._encode_text, augment=False),
                text,
                self.not_augmented_label_encoding_cache,
            )

    def forward(
        self,
        image: torch.Tensor | bytes,
        labels: list[str],
        inc_sub_labels_dict: dict[str, list[str]],
        exc_sub_labels_dict: dict[str, list[str]] | None = None,
        augment_examples: bool = True,
    ) -> torch.Tensor:
        # run an image classifier parameterized by explicit statements on what each label should include or exclude
        # return a tensor of scores for each label
        # each label must include at least one sub-label, and may exclude any number of sub-labels

        if len(labels) == 0:
            raise ValueError("At least one label must be provided")

        if any([len(sub_labels) == 0 for sub_labels in inc_sub_labels_dict.values()]):
            raise ValueError("Each label must include at least one sub-label")

        image_features = self.encode_image(image)

        exc_sub_labels_dict = {} if exc_sub_labels_dict is None else exc_sub_labels_dict
        # filter out empty excs lists
        exc_sub_labels_dict = {
            label: excs for label, excs in exc_sub_labels_dict.items() if len(excs) > 0
        }

        all_labels, all_labels_to_inds = squish_labels(
            labels, inc_sub_labels_dict, exc_sub_labels_dict
        )
        text_features = self.encode_text(all_labels, augment=augment_examples)

        scores = (1 + image_features @ text_features.T) / 2

        label_to_ind = {label: i for i, label in enumerate(labels)}

        pos_labels_to_master_inds, pos_labels_list = zip(
            *[
                v
                for label, incs in inc_sub_labels_dict.items()
                for v in zip([label_to_ind[label]] * len(incs), incs)
            ]
        )
        pos_labels_inds = [all_labels_to_inds[label] for label in pos_labels_list]

        pos_scores = scores[:, pos_labels_inds]

        # pos_labels_to_master_inds indicates which indices we should be taking the max over for each label
        # since our scatter_max will be batched, we need to offset this for each image
        num_labels = len(labels)
        num_images = image_features.shape[0]
        num_incs = len(pos_labels_to_master_inds)
        offsets = (
            torch.arange(num_images).unsqueeze(1).expand(-1, num_incs).flatten()
            * num_labels
        )
        offsets = offsets.to(self.device)
        indices_for_max = (
            torch.tensor(pos_labels_to_master_inds).to(self.device).repeat(num_images)
            + offsets
        )

        max_pos_scores_flat, _ = scatter_max(
            pos_scores.view(-1), indices_for_max, dim_size=num_images * num_labels
        )
        max_pos_scores = max_pos_scores_flat.view(num_images, num_labels)

        # compute the same for the negative labels, if any
        if len(exc_sub_labels_dict) > 0:
            neg_labels_to_master_inds, neg_labels_list = zip(
                *[
                    v
                    for label, excs in exc_sub_labels_dict.items()
                    for v in zip([label_to_ind[label]] * len(excs), excs)
                ]
            )
            neg_labels_inds = [all_labels_to_inds[label] for label in neg_labels_list]

            neg_scores = scores[:, neg_labels_inds]

            num_excs = len(neg_labels_to_master_inds)
            offsets = (
                torch.arange(num_images).unsqueeze(1).expand(-1, num_excs).flatten()
                * num_labels
            )
            offsets = offsets.to(self.device)
            indices_for_max = (
                torch.tensor(neg_labels_to_master_inds)
                .to(self.device)
                .repeat(num_images)
                + offsets
            )

            max_neg_scores_flat, _ = scatter_max(
                neg_scores.view(-1), indices_for_max, dim_size=num_images * num_labels
            )
            max_neg_scores = max_neg_scores_flat.view(num_images, num_labels)

            raw_scores = torch.where(
                max_pos_scores > max_neg_scores, max_pos_scores, 1 - max_neg_scores
            )
        else:
            raw_scores = max_pos_scores

        return raw_scores
