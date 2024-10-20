import torch
from typing import Callable
import numpy as np
from PIL import Image
import io
from lru import LRU


def batch_encode_cache_missed_list_elements(
    encode_fn: Callable[[list], torch.Tensor], args_list: list, cache: dict | LRU | None
) -> torch.Tensor:
    if len(args_list) == 0:
        raise ValueError("args_list should not be empty")

    # NOTE: the batch size may be larger than the cache size
    # NOTE: by passing cache=None, we can disable caching, in which case this is just a straight-through operation

    if cache is not None:
        # first we retrieve any cached values
        cache_hit_inds = []
        cache_miss_inds = []
        cache_hits_tensor_list = []
        cache_misses = []
        for i, arg in enumerate(args_list):
            if (tensor := cache.get(arg)) is not None:
                cache_hit_inds.append(i)
                cache_hits_tensor_list.append(tensor)
            else:
                cache_miss_inds.append(i)
                cache_misses.append(arg)

        cache_hits_tensor = (
            torch.stack(cache_hits_tensor_list, dim=0)
            if len(cache_hits_tensor_list) > 0
            else None
        )
    else:
        cache_hit_inds = []
        cache_miss_inds = list(range(len(args_list)))
        cache_hits_tensor = None
        cache_misses = args_list

    # then we batch encode any cache misses
    if len(cache_misses) > 0:
        cache_misses_tensor = encode_fn(cache_misses)
        for i, arg in enumerate(cache_misses):
            if cache is not None:
                cache[arg] = cache_misses_tensor[i]
    else:
        cache_misses_tensor = None

    # now we merge the cache hits and cache misses
    # such that the order of the output tensors matches the order of the input args
    # NOTE: at least one of cache_hit_inds and cache_miss_inds will be non-empty
    # NOTE: we assume device, dtype, and shape are unchanged between different calls to encode_fn
    if cache_hits_tensor is None:
        output_tensor = cache_misses_tensor
    elif cache_misses_tensor is None:
        output_tensor = cache_hits_tensor
    else:
        output_tensor = torch.empty(
            len(args_list),
            *cache_misses_tensor.shape[1:],
            dtype=cache_misses_tensor.dtype,
            device=cache_misses_tensor.device,
        )
        output_tensor[cache_hit_inds] = cache_hits_tensor
        output_tensor[cache_miss_inds] = cache_misses_tensor

    assert isinstance(output_tensor, torch.Tensor)  # just to make mypy happy

    return output_tensor


def resize_pil_image(
    pil_image: Image.Image, image_size: tuple[int, int]
) -> Image.Image:
    if pil_image.format == "JPEG":
        # try requesting a format-specific conversion
        # this significantly speeds up the subsequent resize operation
        # note that torchvision does NOT try this internally, and is therefore much slower
        # (plus likely using draft here leads to a more accurate resize operation)
        pil_image.draft("RGB", image_size)
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(image_size, Image.BICUBIC)
    return pil_image


def image_bytes_to_tensor(image: bytes, image_size: tuple[int, int]) -> torch.Tensor:
    image_buffer = io.BytesIO(image)
    pil_image = Image.open(image_buffer)
    pil_image = resize_pil_image(pil_image, image_size)
    np_image = np.asarray(pil_image)
    tensor = torch.tensor(np_image).permute(2, 0, 1).unsqueeze(0)
    return tensor


def squish_labels(
    labels: list[str],
    inc_sub_labels_dict: dict[str, list[str]],
    exc_sub_labels_dict: dict[str, list[str]],
) -> tuple[list[str], dict[str, int]]:
    # build one list of labels to encode, without duplicates
    # and lists / dicts containing the indices of each label
    # and the indices of each label's sub-labels
    all_labels_to_inds: dict[str, int] = {}
    all_labels = []

    for label in labels:
        inc_subs = inc_sub_labels_dict.get(label)
        if inc_subs is not None:
            for inc_sub in inc_subs:
                if inc_sub not in all_labels_to_inds:
                    all_labels_to_inds[inc_sub] = len(all_labels_to_inds)
                    all_labels.append(inc_sub)

        exc_subs = exc_sub_labels_dict.get(label)
        if exc_subs is not None:
            for exc_sub in exc_subs:
                if exc_sub not in all_labels_to_inds:
                    all_labels_to_inds[exc_sub] = len(all_labels_to_inds)
                    all_labels.append(exc_sub)

    return all_labels, all_labels_to_inds
