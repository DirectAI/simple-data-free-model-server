import unittest
import torch
from typing_extensions import ClassVar

from modeling.image_classifier import ZeroShotImageClassifierWithFeedback
from modeling.tensor_utils import image_bytes_to_tensor


class TestImageClassifier(unittest.TestCase):
    # we have to define these here because mypy doesn't dive into the init hiding behind the classmethod
    image_classifier = (
        NotImplemented
    )  # type: ClassVar[ZeroShotImageClassifierWithFeedback]
    coke_bottle_image_bytes = NotImplemented  # type: ClassVar[bytes]
    coke_can_image_bytes = NotImplemented  # type: ClassVar[bytes]
    default_labels = NotImplemented  # type: ClassVar[list[str]]
    default_incs = NotImplemented  # type: ClassVar[dict[str, list[str]]]
    default_excs = NotImplemented  # type: ClassVar[dict[str, list[str]]]

    # we are going to consider out-of-scope testing different base models and datasets
    # which means we can just initialize the model and reuse it for all tests
    @classmethod
    def setUpClass(cls) -> None:
        cls.image_classifier = ZeroShotImageClassifierWithFeedback(
            base_model_name="ViT-H-14-quickgelu",
            dataset_name="dfn5b",
            max_text_batch_size=256,
            max_image_batch_size=256,
            device="cuda:0",
            lru_cache_size=4096,
            jit=False,  # useful to speed up unit testing since throughput is not a concern, just start-up time
            fp16=True,
        )

        coke_bottle_filepath = "unit_tests/sample_data/coke_through_the_ages.jpeg"
        with open(coke_bottle_filepath, "rb") as f:
            cls.coke_bottle_image_bytes = f.read()
        coke_can_filepath = "unit_tests/sample_data/coke_can.jpg"
        with open(coke_can_filepath, "rb") as f:
            cls.coke_can_image_bytes = f.read()

        cls.default_labels = ["can", "bottle", "plate", "spoon"]
        cls.default_incs = {
            "can": ["can", "soda can", "aluminum can"],
            "bottle": ["bottle", "glass bottle", "plastic bottle", "water bottle"],
            "plate": ["plate", "dinner plate"],
            "spoon": ["spoon", "metal spoon", "plastic spoon"],
        }
        cls.default_excs = {
            "plate": ["bowl", "saucer"],
            "spoon": [
                "fork",
            ],
        }

    def test_classify_image_bytes(self) -> None:
        with torch.no_grad():
            raw_scores = self.image_classifier(
                self.coke_bottle_image_bytes,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=self.default_excs,
            )

        self.assertEqual(raw_scores.shape, (1, len(self.default_labels)))
        pred_label = self.default_labels[raw_scores[0].argmax()]
        self.assertEqual(pred_label, "bottle")

    def test_classify_image_tensor(self) -> None:
        with torch.no_grad():
            image_tensor = image_bytes_to_tensor(
                self.coke_bottle_image_bytes, self.image_classifier.img_size
            )
            raw_scores = self.image_classifier(
                image_tensor,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=self.default_excs,
            )

        self.assertEqual(raw_scores.shape, (1, len(self.default_labels)))
        pred_label = self.default_labels[raw_scores[0].argmax()]
        self.assertEqual(pred_label, "bottle")

    def test_classify_image_batch(self) -> None:
        with torch.no_grad():
            coke_bottle_image_tensor = image_bytes_to_tensor(
                self.coke_bottle_image_bytes, self.image_classifier.img_size
            )
            coke_can_image_tensor = image_bytes_to_tensor(
                self.coke_can_image_bytes, self.image_classifier.img_size
            )
            image_batch = torch.cat(
                [
                    coke_bottle_image_tensor,
                ]
                * 4
                + [
                    coke_can_image_tensor,
                ]
                * 4,
                dim=0,
            )
            raw_scores = self.image_classifier(
                image_batch,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=self.default_excs,
            )

        self.assertEqual(raw_scores.shape, (8, len(self.default_labels)))
        pred_labels = [
            self.default_labels[raw_score.argmax()] for raw_score in raw_scores
        ]
        self.assertEqual(
            pred_labels,
            ["bottle", "bottle", "bottle", "bottle", "can", "can", "can", "can"],
        )

    def test_random_tensor_classification_in_batch(self) -> None:
        random_tensors = torch.rand(
            128, 3, *self.image_classifier.img_size, device=self.image_classifier.device
        )
        bs_1_raw_scores_list = []
        with torch.no_grad():
            for random_tensor in random_tensors:
                raw_scores = self.image_classifier(
                    random_tensor.unsqueeze(0),
                    labels=self.default_labels,
                    inc_sub_labels_dict=self.default_incs,
                    exc_sub_labels_dict=self.default_excs,
                )
                bs_1_raw_scores_list.append(raw_scores)

            bs_1_raw_scores = torch.cat(bs_1_raw_scores_list, dim=0)

            single_batch_raw_scores = self.image_classifier(
                random_tensors,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=self.default_excs,
            )

        # results aren't necessarily consistent across different batch sizes
        # see https://discuss.pytorch.org/t/batch-size-changes-linear-layer-output-values/143706
        # the discrepancy increases with lower bit-widths
        # in fp16 it's on the order of 5e-4, with fp32 it's on the order of 2e-7
        # we use fp16 by default, so we'll use the higher threshold
        max_diff = (bs_1_raw_scores - single_batch_raw_scores).abs().max().item()
        self.assertTrue(max_diff < 5e-4)
