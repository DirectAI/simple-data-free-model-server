import torch
from typing import Callable
import numpy as np
from PIL import Image
import io
from lru import LRU


def batch_encode_cache_missed_list_elements(encode_fn: Callable[[list], torch.Tensor], args_list: list, cache: dict|LRU|None) -> torch.Tensor:
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
                
        cache_hits_tensor = torch.stack(cache_hits_tensor_list, dim=0) if len(cache_hits_tensor_list) > 0 else None
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
        output_tensor = torch.empty(len(args_list), *cache_misses_tensor.shape[1:], dtype=cache_misses_tensor.dtype, device=cache_misses_tensor.device)
        output_tensor[cache_hit_inds] = cache_hits_tensor
        output_tensor[cache_miss_inds] = cache_misses_tensor
    
    return output_tensor


def image_bytes_to_tensor(image: bytes, image_size: tuple[int, int]) -> torch.Tensor:
    image_buffer = io.BytesIO(image)
    pil_image = Image.open(image_buffer)
    if pil_image.format == 'JPEG':
        # try requesting a format-specific conversion
        # this significantly speeds up the subsequent resize operation
        # note that torchvision does NOT try this internally, and is therefore much slower
        # (plus likely using draft here leads to a more accurate resize operation)
        pil_image.draft("RGB", image_size)
    pil_image = pil_image.convert('RGB')
    pil_image = pil_image.resize(image_size, Image.BICUBIC)
    np_image = np.asarray(pil_image)
    tensor = torch.tensor(np_image).permute(2, 0, 1).unsqueeze(0)
    return tensor
    