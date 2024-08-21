import unittest
import torch
from typing_extensions import ClassVar

from modeling.object_detector import (
    ZeroShotObjectDetectorWithFeedback,
    created_padded_tensor_from_bytes,
)


class TestObjectDetector(unittest.TestCase):
    # we have to define these here because mypy doesn't dive into the init hiding behind the classmethod
    object_detector = NotImplemented  # type: ClassVar[ZeroShotObjectDetectorWithFeedback]
    coke_bottle_image_bytes = NotImplemented  # type: ClassVar[bytes]
    default_labels = NotImplemented  # type: ClassVar[list[str]]
    default_incs = NotImplemented  # type: ClassVar[dict[str, list[str]]]
    default_excs = NotImplemented  # type: ClassVar[dict[str, list[str]]]
    default_nms_thre = NotImplemented  # type: ClassVar[float]
    default_conf_thres = NotImplemented  # type: ClassVar[dict[str, float]]
    
    @classmethod
    def setUpClass(cls) -> None:
        cls.object_detector = ZeroShotObjectDetectorWithFeedback()
        
        coke_bottle_filepath = "unit_tests/sample_data/coke_through_the_ages.jpeg"
        with open(coke_bottle_filepath, "rb") as f:
            cls.coke_bottle_image_bytes = f.read()
        
        cls.default_labels = ["bottle", "moose"]
        cls.default_incs = {
            "bottle": ["bottle", "glass bottle", "plastic bottle", "water bottle"],
            "moose": ["moose", "elk", "deer"],
        }
        cls.default_excs = {
            "bottle": ["can", "soda can", "aluminum can"],
        }
        cls.default_nms_thre = 0.4
        cls.default_conf_thres = {
            "bottle": 0.1,
            "moose": 0.1,
        }
    
    def test_detect_objects_from_image_bytes(self) -> None:
        with torch.no_grad():
            batched_predicted_boxes = self.object_detector(
                self.coke_bottle_image_bytes,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=self.default_excs,
                nms_thre=self.default_nms_thre,
                label_conf_thres=self.default_conf_thres,
            )
            
            print(batched_predicted_boxes)