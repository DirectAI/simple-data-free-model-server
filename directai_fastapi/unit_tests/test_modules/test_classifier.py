import unittest
import torch

from modeling.image_classifier import ZeroShotImageClassifierWithFeedback
from modeling.tensor_utils import image_bytes_to_tensor


class TestImageClassifier(unittest.TestCase):
    # we are going to consider out-of-scope testing different base models and datasets
    # which means we can just initialize the model and reuse it for all tests
    def setUp(self) -> None:
        self.image_classifier = ZeroShotImageClassifierWithFeedback(
            base_model_name='ViT-H-14-quickgelu',
            dataset_name='dfn5b',
            max_text_batch_size=256,
            max_image_batch_size=256,
            device='cuda:0',
            lru_cache_size=4096,
            jit=False  # useful to speed up unit testing since throughput is not a concern, just start-up time
        )
        
        coke_bottle_filepath = "unit_tests/sample_data/coke_through_the_ages.jpeg"
        with open(coke_bottle_filepath, "rb") as f:
            self.coke_bottle_image_bytes = f.read()
        coke_can_filepath = "unit_tests/sample_data/coke_can.jpg"
        with open(coke_can_filepath, "rb") as f:
            self.coke_can_image_bytes = f.read()
    
        self.default_labels = ["can", "bottle", "plate", "spoon"]
        self.default_incs = {
            "can": ["can", "soda can", "aluminum can"],
            "bottle": ["bottle", "glass bottle", "plastic bottle", "water bottle"],
            "plate": ["plate", "dinner plate"],
            "spoon": ["spoon", "metal spoon", "plastic spoon"]
        }
        self.default_excs = {
            "plate": ["bowl", "saucer"],
            "spoon": ["fork",]
        }
    
    def test_classify_image_bytes(self) -> None:
        with torch.no_grad():
            raw_scores = self.image_classifier(
                self.coke_bottle_image_bytes,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=self.default_excs
            )
        
        self.assertEqual(raw_scores.shape, (1, len(self.default_labels)))
        pred_label = self.default_labels[raw_scores[0].argmax()]
        self.assertEqual(pred_label, 'bottle')
    
    def test_classify_image_tensor(self) -> None:
        with torch.no_grad():
            image_tensor = image_bytes_to_tensor(self.coke_bottle_image_bytes, self.image_classifier.img_size)
            raw_scores = self.image_classifier(
                image_tensor,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=self.default_excs
            )
        
        self.assertEqual(raw_scores.shape, (1, len(self.default_labels)))
        pred_label = self.default_labels[raw_scores[0].argmax()]
        self.assertEqual(pred_label, 'bottle')
    
    def test_classify_image_batch(self) -> None:
        with torch.no_grad():
            coke_bottle_image_tensor = image_bytes_to_tensor(self.coke_bottle_image_bytes, self.image_classifier.img_size)
            coke_can_image_tensor = image_bytes_to_tensor(self.coke_can_image_bytes, self.image_classifier.img_size)
            image_batch = torch.cat([coke_bottle_image_tensor,] * 4 + [coke_can_image_tensor,] * 4, dim=0)
            raw_scores = self.image_classifier(
                image_batch,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=self.default_excs
            )
        
        self.assertEqual(raw_scores.shape, (8, len(self.default_labels)))
        pred_labels = [self.default_labels[raw_score.argmax()] for raw_score in raw_scores]
        self.assertEqual(pred_labels, ['bottle', 'bottle', 'bottle', 'bottle', 'can', 'can', 'can', 'can'])