from typing import List, Optional, Tuple
import torch
from PIL import Image
import torch
from torch import nn
import torchvision
import numpy as np
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Owlv2VisionModel
from transformers.models.owlv2.modeling_owlv2 import Owlv2Attention
import time
from typing import Union
from torch_scatter import scatter_max
from flash_attn import flash_attn_func
import io
from lru import LRU
from functools import partial

from modeling.prompt_templates import medium_hypothesis_formats, noop_hypothesis_formats
from modeling.tensor_utils import (
    batch_encode_cache_missed_list_elements,
    resize_pil_image,
    squish_labels,
)


def created_padded_tensor_from_bytes(
    image_bytes: bytes, image_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_image_tensor = torch.ones((1, 3, *image_size)) * 114.0

    # TODO: add nonblocking streaming to GPU
    
    image_buffer = io.BytesIO(image_bytes)
    pil_image = Image.open(image_buffer)
    current_size = pil_image.size
    
    r = min(image_size[0] / current_size[0], image_size[1] / current_size[1])
    target_size = (int(r * current_size[0]), int(r * current_size[1]))
    
    pil_image = resize_pil_image(pil_image, target_size)

    np_image = np.asarray(pil_image)
    torch_image = torch.tensor(np_image).permute(2, 0, 1).unsqueeze(0)
    
    padded_image_tensor[:, :, :torch_image.shape[2], :torch_image.shape[3]] = torch_image
    
    image_scale_ratios = torch.tensor([r,])
    
    return padded_image_tensor, image_scale_ratios
    

def flash_attn_owl_vit_encoder_forward(
        self: Owlv2Attention,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> tuple[torch.Tensor, None]:
    assert not output_attentions, "output_attentions not supported for flash attention implementation"
    assert attention_mask is None, "attention_mask not supported for flash attention implementation"
    # technically flash_attn DOES support causal attention
    # but the OWL usage of causal attention mask does not limit it to true causal attention
    # we don't support generalized attention, so we're just going to assert causal attention mask is ALSO None
    assert causal_attention_mask is None, "causal_attention_mask not supported for flash attention implementation"

    bsz, tgt_len, embed_dim = hidden_states.shape
    
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    query_states = query_states.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim)
    key_states = key_states.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim)
    value_states = value_states.contiguous().view(bsz, tgt_len, self.num_heads, self.head_dim)
    
    # convert to appropriate dtype
    # NOTE: bf16 may be more appropriate than fp16
    query_states = query_states.to(torch.float16)
    key_states = key_states.to(torch.float16)
    value_states = value_states.to(torch.float16)
    
    attn_output = flash_attn_func(
        query_states,
        key_states,
        value_states,
        dropout_p=0,
        softmax_scale=self.scale,
    )
    
    attn_output = attn_output.view(bsz, tgt_len, embed_dim)
    
    # convert back to appropriate dtype
    attn_output = attn_output.to(hidden_states.dtype)
    
    attn_output = self.out_proj(attn_output)
    
    return attn_output, None
    

# we're copying the function signature from the original
# and just replacing the method with a faster one based on flash_attn
# we could subclass the original, but that would require us to subclass the entire model
# so we're just going to monkey patch it, as the output should be identical with the same inputs
# Owlv2Attention.forward = flash_attn_owl_vit_encoder_forward
# for owlv2_vision_model_encoder_layer in Owlv2VisionModel.vision_model.encoder.layers:
#     owlv2_vision_model_encoder_layer.self_attn.forward = flash_attn_owl_vit_encoder_forward


class VisionModelWrapper(nn.Module):
    def __init__(self, vision_model: Owlv2VisionModel) -> None:
        super().__init__()
        
        self.vision_model = vision_model
        
        # we're going to monkey patch the forward method of the attention layers
        # to replace it with a faster one based on flash_attn
        # the alternative is to subclass the entire model, but that's a lot of work
        # so we're just going to define a replacement with the same function signature
        # and assert that the input is as expected
        for owlv2_vision_model_encoder_layer in self.vision_model.encoder.layers:
            owlv2_vision_model_encoder_layer.self_attn.forward = partial(
                flash_attn_owl_vit_encoder_forward, owlv2_vision_model_encoder_layer.self_attn
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.vision_model(pixel_values=image, return_dict=True)
        
        # Get image embedding
        last_hidden_states = vision_outputs[0]
        image_embeds = self.vision_model.post_layernorm(last_hidden_states)
        
        return image_embeds


class WrappedImageEmbedder(nn.Module):
    def __init__(self, model: Owlv2ForObjectDetection) -> None:
        super().__init__()
        
        self.model = model
        self.wrapped_vision_model = VisionModelWrapper(self.model.owlv2.vision_model)
    
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_embeds = self.wrapped_vision_model(image)
        
        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.model.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        feature_map = image_embeds.reshape(new_size)
        
        # Get class head features
        # we do dot prod between image_class_embeds and query embeddings, then do (pred + shift) * scale
        image_class_embeds = self.model.class_head.dense0(image_embeds)
        image_class_embeds = image_class_embeds / image_class_embeds.norm(dim=-1, keepdim=True, p=2)
        logit_shift = self.model.class_head.logit_shift(image_embeds)
        logit_scale = self.model.class_head.logit_scale(image_embeds)
        logit_scale = self.model.class_head.elu(logit_scale) + 1
        
        # Get box head features
        # NOTE: this is in a specific format, handle later
        pred_boxes = self.model.box_predictor(image_embeds, feature_map)
        
        # filter out patches that are unlikely to map to objects
        # the paper takes the top 10% during training, but we'll take the top 300 to be more in line with DETR
        objectness_scores = self.model.objectness_predictor(image_embeds)
        # compute the top 300 objectness indices
        indices = torch.topk(objectness_scores, 300, dim=1).indices
        # filter all the other stuff
        image_class_embeds = image_class_embeds.gather(1, indices.unsqueeze(-1).expand(-1, -1, image_class_embeds.shape[-1]))
        logit_shift = logit_shift.gather(1, indices.unsqueeze(-1).expand(-1, -1, logit_shift.shape[-1]))
        logit_scale = logit_scale.gather(1, indices.unsqueeze(-1).expand(-1, -1, logit_scale.shape[-1]))
        pred_boxes = pred_boxes.gather(1, indices.unsqueeze(-1).expand(-1, -1, pred_boxes.shape[-1]))
        
        assert image_class_embeds.shape[1] == 300
                
        return image_class_embeds, logit_shift, logit_scale, pred_boxes


class ZeroShotObjectDetectorWithFeedback(nn.Module):
    def __init__(
        self,
        model_name: str = "google/owlv2-large-patch14-ensemble",
        image_size: tuple[int, int] = (1008, 1008),
        device: torch.device | str = "cuda",
        lru_cache_size: int = 4096,
        jit: bool = True,
    ):
        super().__init__()
        
        self.device = device
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()
        self.processor = Owlv2Processor.from_pretrained(model_name)
        
        if jit:
            self.wrapped_image_embedder = torch.jit.trace_module(
                WrappedImageEmbedder(self.model),
                {"forward": (torch.randn(1, 3, *image_size, device=device),)},
            )
        else:
            self.wrapped_image_embedder = WrappedImageEmbedder(self.model)

        # we cache the text embeddings to avoid recomputing them
        # we use an LRU cache to avoid running out of memory
        # especially because likely the tensors will be large and stored in GPU memory
        self.augmented_label_encoding_cache: LRU | None = (
            LRU(lru_cache_size) if lru_cache_size > 0 else None
        )
        self.not_augmented_label_encoding_cache: LRU | None = (
            LRU(lru_cache_size) if lru_cache_size > 0 else None
        )
        
        self.image_size = image_size
        self.rgb_means = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.rgb_stds = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def _encode_text(self, text: list[str], augment: bool = True) -> torch.Tensor:
        # NOTE: object detector liturature tends to use fewer templates than image classifiers
        templates = medium_hypothesis_formats if augment else noop_hypothesis_formats
        augmented_text = [template.format(t) for t in text for template in templates]
        
        processor_output = self.processor(text=augmented_text, return_tensors="pt", padding=True, truncation=True)
        input_ids = processor_output.input_ids.to(self.device)
        attn_mask = processor_output.attention_mask.to(self.device)
        
        # TODO: add appropriate batching to avoid OOM
        text_output = self.model.owlv2.text_model(
            input_ids=input_ids, attention_mask=attn_mask, return_dict=True
        )
            
        embeddings = text_output[1]
        embeddings = self.model.owlv2.text_projection(embeddings)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True, p=2)
        
        embeddings = embeddings.reshape(len(text), len(templates), embeddings.shape[1])
        embeddings = embeddings.mean(dim=1)
        embeddings = embeddings / embeddings.norm(dim=1, keepdim=True, p=2)
        
        return embeddings
    
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

    def get_image_data(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        # we do the normalization here to make sure we have access to the right parameters
        image = image / 255.0
        image = (image - self.rgb_means) / self.rgb_stds
        
        image_class_embeds, logit_shift, logit_scale, pred_boxes = self.wrapped_image_embedder(image)
        
        return {
            "image_class_embeds": image_class_embeds,
            "logit_shift": logit_shift,
            "logit_scale": logit_scale,
            "pred_boxes": pred_boxes,
        }

    def forward(
        self,
        image: torch.Tensor | bytes,
        labels: list[str],
        inc_sub_labels_dict: dict[str, list[str]],
        exc_sub_labels_dict: dict[str, list[str]] | None = None,
        label_conf_thres: dict[str, float] | None = None,
        augment_examples: bool = True,
        nms_thre: float = 0.4,
        run_class_agnostic_nms: bool = False,
        image_scale_ratios: torch.Tensor | None = None,
    ) -> list[list[torch.Tensor]]:
        assert not run_class_agnostic_nms, "Class-agnostic NMS not yet implemented"
        
        if isinstance(image, bytes):
            assert image_scale_ratios is None, "image_scale_ratios must be None if image is bytes as we define the scale internally"
            image_tensor, image_scale_ratios = created_padded_tensor_from_bytes(image, self.image_size)
        else:
            assert image_scale_ratios is not None, "image_scale_ratios must be provided if image is a tensor as we cannot derive the scale internally"
            image_tensor = image
        
        if label_conf_thres is None:
            label_conf_thres = {}

        if len(labels) == 0:
            raise ValueError("At least one label must be provided")

        if any([len(sub_labels) == 0 for sub_labels in inc_sub_labels_dict.values()]):
            raise ValueError("Each label must include at least one sub-label")

        image_tensor = image_tensor.to(self.device)

        image_data = self.get_image_data(image_tensor)

        exc_sub_labels_dict = {} if exc_sub_labels_dict is None else exc_sub_labels_dict
        # filter out empty excs lists
        exc_sub_labels_dict = {
            label: excs for label, excs in exc_sub_labels_dict.items() if len(excs) > 0
        }

        all_labels, all_labels_to_inds = squish_labels(
            labels, inc_sub_labels_dict, exc_sub_labels_dict
        )
        text_features = self.encode_text(all_labels, augment=augment_examples)

        scores_by_image_and_box = compute_query_fit(
            text_features,
            image_data["image_class_embeds"],
            image_data["logit_shift"],
            image_data["logit_scale"],
        )
        # NOTE that scores_by_image_and_box is of shape [num_images, num_boxes, len(all_labels)]
        # for the extracting of the per-box pro and con scores, we don't care about differentiating the first two dimensions
        # so we flatten them to make the scatter_max operation easier
        # and then we reshape them back to the original shape
        scores = scores_by_image_and_box.view(-1, len(all_labels))
        # now we can proceed in the same way as the image classifier

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
        # since our scatter_max will be batched, we need to offset this for each box
        num_labels = len(labels)
        num_boxes = scores.shape[0]
        num_incs = len(pos_labels_to_master_inds)
        offsets = (
            torch.arange(num_boxes).unsqueeze(1).expand(-1, num_incs).flatten()
            * num_labels
        )
        offsets = offsets.to(self.device)
        indices_for_max = (
            torch.tensor(pos_labels_to_master_inds).to(self.device).repeat(num_boxes)
            + offsets
        )

        max_pos_scores_flat, _ = scatter_max(
            pos_scores.view(-1), indices_for_max, dim_size=num_boxes * num_labels
        )
        max_pos_scores = max_pos_scores_flat.view(num_boxes, num_labels)

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
                torch.arange(num_boxes).unsqueeze(1).expand(-1, num_excs).flatten()
                * num_labels
            )
            offsets = offsets.to(self.device)
            indices_for_max = (
                torch.tensor(neg_labels_to_master_inds)
                .to(self.device)
                .repeat(num_boxes)
                + offsets
            )

            max_neg_scores_flat, _ = scatter_max(
                neg_scores.view(-1), indices_for_max, dim_size=num_boxes * num_labels
            )
            max_neg_scores = max_neg_scores_flat.view(num_boxes, num_labels)
        else:
            # if we have no negative labels, we just set the max neg scores to zero
            # NOTE: possible to speed things up by skipping the ops conditional on having negative labels
            max_neg_scores = torch.zeros_like(max_pos_scores)
        
        # now reshape the scores to [num_images, num_boxes, num_labels]
        max_pos_scores = max_pos_scores.view(image_data["pred_boxes"].shape[0], image_data["pred_boxes"].shape[1], num_labels)
        max_neg_scores = max_neg_scores.view(image_data["pred_boxes"].shape[0], image_data["pred_boxes"].shape[1], num_labels)
        
        # unlike the image classifier, we have to suppress boxes based on the scores of their neighbors
        # we do this via a modified NMS algorithm
        # because it operates over a variable-sized graph of boxes, it's hard to vectorize
        # so we dump it into a script function that does fork-based async processing
        # the output is a per-image list of per-object boxes in tlbr-score format
        batched_predicted_boxes = batched_run_nms_based_box_suppression_for_all_objects(
            max_pos_scores,
            max_neg_scores,
            image_data["pred_boxes"],
            image_tensor.shape[2] / image_scale_ratios,
            torch.tensor([label_conf_thres.get(label, 0.0) for label in labels], device=self.device),
            nms_thre,
            run_class_agnostic_nms,
        )
        
        return batched_predicted_boxes


@torch.jit.script
def compute_query_fit(
    query_embeds: torch.Tensor,
    image_class_embeds: torch.Tensor,
    logit_shift: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    # Compute query fit
    pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)
    pred_logits = (pred_logits + logit_shift) * logit_scale
    
    return torch.sigmoid(pred_logits)


@torch.jit.script
def compute_iou_adjacency_list(boxes: torch.Tensor, nms_thre: float) -> list[torch.Tensor]:
    boxes = boxes.clone()
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    ious = torchvision.ops.box_iou(boxes, boxes)
    ious = ious >= nms_thre
    # Set diagonal elements to zero .. no self loops!
    ious.fill_diagonal_(0)
    
    edges = torch.nonzero(ious).unbind(-1)
    
    return edges


@torch.jit.script
def find_in_sorted_tensor(sorted_tensor: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.searchsorted(sorted_tensor, query)
    indices_clamped = torch.clamp(indices, max=sorted_tensor.size(0) - 1)
    present = sorted_tensor[indices_clamped] == query
    in_bounds = indices < sorted_tensor.size(0)
    found_mask = present & in_bounds
    return found_mask, indices


@torch.jit.script
def compute_candidate_nms_via_adjacency_list(pro_max: torch.Tensor, con_max: torch.Tensor, adjacency_list: list[torch.Tensor], conf_thre: float) -> torch.Tensor:
    # we use a scatter_max to efficiently compute, for each bounding box, the max con score of its adjacent boxes
    expanded_con_max = con_max[adjacency_list[0]]
    adjacent_con_max = scatter_max(expanded_con_max, adjacency_list[1], dim_size=con_max.shape[0])[0]
    # we then filter down to boxes that both exceed the confidence threshold and are not suppressed by negative examples
    # we do this by filtering on three expressions:
    # 1. pro_max >= conf_thre: the box has a high enough confidence
    # 2. pro_max >= adjacent_con_max: the box has a higher confidence than any adjacent boxes have negative confidence
    # 3. pro_max >= con_max: the box has a higher confidence than its own negative confidence
    # NOTE: could make this more efficient perhaps by filtering out the easy ones prior to the scatter_max
    pro_valid = (pro_max >= conf_thre) * (pro_max >= adjacent_con_max) * (pro_max >= con_max)
    pro_valid_inds = pro_valid.nonzero().squeeze(1)
        
    if pro_valid_inds.numel() == 0 or adjacency_list[0].numel() == 0:
        # no boxes are valid or no boxes have any overlap with any other boxes
        # either way, we can skip the NMS step
        return pro_valid_inds

    # remove reported overlaps with boxes that are not valid
    # this shrinks the graph that we need to do NMS over
    first_node_valid, _ = find_in_sorted_tensor(pro_valid_inds, adjacency_list[0])
    second_node_valid, _ = find_in_sorted_tensor(pro_valid_inds, adjacency_list[1])

    nms_inds = torch.nonzero(first_node_valid * second_node_valid).squeeze(1)
    modified_adjacency_list = [adjacency_list[0][nms_inds], adjacency_list[1][nms_inds]]
    
    if nms_inds.numel() == 0:
        # none of the remaining boxes have any overlap with any other remaining boxes
        # so we can skip the NMS step
        survive_inds = pro_valid.nonzero().squeeze(1)
        return survive_inds

    # we compute the indices of the start and end of each box's adjacent boxes
    # since our graph representation is just a list of edges, we would like to know which edges correspond to which nodes
    # as the first node is sorted, we can just take the difference between adjacent nodes to get the start and end of each node's edges
    first_node = modified_adjacency_list[0]
    zero_tensor = torch.tensor([0], device=pro_max.device)
    change_inds = (first_node[1:] != first_node[:-1]).nonzero()[:, 0]
    len_tensor = torch.tensor([first_node.shape[0]], device=pro_max.device)
    inds_of_adj_boxes = torch.cat([zero_tensor, change_inds + 1, len_tensor])
    
    # we then run a NMS over the (remaining) boxes
    sorted_pro_valid_inds = pro_valid_inds[pro_max[pro_valid].argsort(descending=True)]
        
    # check which boxes have graph connections
    unique_nodes = first_node[inds_of_adj_boxes[:-1]]
    has_connection, graph_node_indices = find_in_sorted_tensor(unique_nodes, sorted_pro_valid_inds)
    connected_sorted_pro_valid_inds = sorted_pro_valid_inds[has_connection]
    graph_indices = graph_node_indices[has_connection]
        
    for i, j in zip(connected_sorted_pro_valid_inds, graph_indices):
        if pro_valid[i] == 0:
            continue
        
        remapped_start_ind = inds_of_adj_boxes[j]
        remapped_end_ind = inds_of_adj_boxes[j+1]
        adj_boxes = modified_adjacency_list[1][remapped_start_ind:remapped_end_ind]
        pro_valid[adj_boxes] = 0
    
    survive_inds = pro_valid.nonzero().squeeze(1)
        
    return survive_inds


@torch.jit.script
def run_nms_based_box_suppression_for_one_object(
    pro_max: torch.Tensor,
    con_max: torch.Tensor,
    pred_boxes: torch.Tensor,
    adjacency_list: list[torch.Tensor],
    image_scale: float,
    conf_thre: float = 0.001,
) -> torch.Tensor:
    survive_inds = compute_candidate_nms_via_adjacency_list(pro_max, con_max, adjacency_list, conf_thre)
    
    boxes = pred_boxes.squeeze(0)[survive_inds]
    logits = pro_max[survive_inds]
    
    logits = logits.unsqueeze(-1)
    boxes = boxes * image_scale
    
    # Convert boxes from center_x, center_y, width, height (cx_cy_w_h) to top_left_x, top_left_y, bottom_right_x, bottom_right_y (tlbr)
    cx, cy, w, h = boxes.unbind(-1)
    tl_x = cx - 0.5 * w
    tl_y = cy - 0.5 * h
    br_x = cx + 0.5 * w
    br_y = cy + 0.5 * h
    boxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=-1)
    
    boxes_with_scores = torch.cat([boxes, logits], dim=-1)
    boxes_with_scores = boxes_with_scores[boxes_with_scores[:, 4].argsort(descending=True)]
    
    print(boxes_with_scores)
    print(boxes_with_scores.shape)
    
    return boxes_with_scores


@torch.jit.script
def run_nms_based_box_suppression_for_all_objects(
    pro_max: torch.Tensor,
    con_max: torch.Tensor,
    pred_boxes: torch.Tensor,
    image_scale: float,
    conf_thres: torch.Tensor,
    nms_thre: float = 0.4,
    run_class_agnostic_nms: bool = False,
) -> list[torch.Tensor]:
    # pred_boxes is assumed to be [num_boxes, 4]
    # pro_max and con_max are assumed to be [num_boxes, num_objects]
    # conf_thres is assumed to be [num_objects]
    adjacency_list = compute_iou_adjacency_list(pred_boxes, nms_thre)
    
    futures = [
        torch.jit.fork(
            run_nms_based_box_suppression_for_one_object,
            pro_max[:, i],
            con_max[:, i],
            pred_boxes,
            adjacency_list,
            image_scale,
            conf_thres[i],
        ) for i in range(pro_max.shape[1])
    ]
    
    predicted_boxes = [torch.jit.wait(fut) for fut in futures]
    
    # TODO: add class-agnostic NMS
    assert not run_class_agnostic_nms, "Class-agnostic NMS not yet implemented"
    
    return predicted_boxes


@torch.jit.script
def batched_run_nms_based_box_suppression_for_all_objects(
    pro_max: torch.Tensor,
    con_max: torch.Tensor,
    pred_boxes: torch.Tensor,
    image_scales: torch.Tensor,
    conf_thres: torch.Tensor,
    nms_thre: float = 0.4,
    run_class_agnostic_nms: bool = False,
) -> list[list[torch.Tensor]]:
    # pred_boxes is assumed to be [num_images, num_boxes, 4]
    # pro_max and con_max are assumed to be [num_images, num_boxes, num_objects]
    # conf_thres is assumed to be [num_objects]
    # image_scales is assumed to be [num_images]
    futures = [
        torch.jit.fork(
            run_nms_based_box_suppression_for_all_objects,
            pro_max[i],
            con_max[i],
            pred_boxes[i],
            image_scales[i].item(),
            conf_thres,
            nms_thre,
            run_class_agnostic_nms,
        ) for i in range(pro_max.shape[0])
    ]
    
    batched_predicted_boxes = [torch.jit.wait(fut) for fut in futures]
    
    return batched_predicted_boxes