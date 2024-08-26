import unittest
import torch
import torchvision  # type: ignore
from typing_extensions import ClassVar

from modeling.object_detector import (
    ZeroShotObjectDetectorWithFeedback,
    created_padded_tensor_from_bytes,
    compute_iou_adjacency_list,
    run_nms_via_adjacency_list,
    run_nms_based_box_suppression_for_all_objects,
)


class TestHelperFunctions(unittest.TestCase):
    def test_nms_via_adjacency_list(self) -> None:
        nms_thre = 0.1
        n_boxes = 1024
        cxcywh_boxes = torch.rand(n_boxes, 4)
        scores = torch.rand(n_boxes)

        tlbr_boxes = cxcywh_boxes.clone()
        tlbr_boxes[:, :2] = cxcywh_boxes[:, :2] - cxcywh_boxes[:, 2:] / 2
        tlbr_boxes[:, 2:] = cxcywh_boxes[:, :2] + cxcywh_boxes[:, 2:] / 2

        box_indices_by_descending_score = torch.argsort(scores, descending=True)
        adjacency_list = compute_iou_adjacency_list(cxcywh_boxes, nms_thre=nms_thre)
        adjacency_survived_inds = run_nms_via_adjacency_list(
            box_indices_by_descending_score, adjacency_list
        )

        torchvision_survived_inds = torchvision.ops.nms(
            tlbr_boxes, scores, iou_threshold=nms_thre
        )

        self.assertTrue(torch.all(adjacency_survived_inds == torchvision_survived_inds))

    def test_class_agnostic_has_no_effect_on_single_class(self) -> None:
        n_boxes = 512
        pro_max = torch.rand(n_boxes, 1)
        con_max = torch.rand(n_boxes, 1)
        cxcywh_boxes = torch.rand(n_boxes, 4)
        conf_thres = torch.tensor([0.1])
        nms_thre = 0.1

        class_believer_boxes = run_nms_based_box_suppression_for_all_objects(
            pro_max,
            con_max,
            cxcywh_boxes,
            1.0,
            conf_thres,
            nms_thre,
            run_class_agnostic_nms=True,
        )
        self.assertEqual(len(class_believer_boxes), 1)

        class_agnostic_boxes = run_nms_based_box_suppression_for_all_objects(
            pro_max,
            con_max,
            cxcywh_boxes,
            1.0,
            conf_thres,
            nms_thre,
            run_class_agnostic_nms=False,
        )

        self.assertTrue(
            torch.all(torch.eq(class_believer_boxes[0], class_agnostic_boxes[0]))
        )

    def test_class_agnostic_has_no_effect_on_multiclass_with_no_overlap(self) -> None:
        n_classes = 4
        n_boxes = 512 * n_classes
        pro_max = torch.zeros(n_boxes, n_classes)
        con_max = torch.zeros(n_boxes, n_classes)
        cxcywh_boxes = torch.rand(n_boxes, 4)
        conf_thres = torch.tensor([0.1] * n_classes)
        nms_thre = 0.1

        # adjust the scores such that each class has nonzero scores in exactly 1/n_classes of the boxes
        # and shift those boxes so that they don't overlap between classes
        for i in range(n_classes):
            pro_max[i::n_classes, i] = torch.rand(n_boxes // n_classes)
            con_max[i::n_classes, i] = torch.rand(n_boxes // n_classes)
            cxcywh_boxes[i::n_classes, 0] += i * 10

        class_believer_boxes = run_nms_based_box_suppression_for_all_objects(
            pro_max,
            con_max,
            cxcywh_boxes,
            1.0,
            conf_thres,
            nms_thre,
            run_class_agnostic_nms=True,
        )
        self.assertEqual(len(class_believer_boxes), n_classes)

        class_agnostic_boxes = run_nms_based_box_suppression_for_all_objects(
            pro_max,
            con_max,
            cxcywh_boxes,
            1.0,
            conf_thres,
            nms_thre,
            run_class_agnostic_nms=False,
        )

        for i in range(n_classes):
            self.assertTrue(
                torch.all(torch.eq(class_believer_boxes[i], class_agnostic_boxes[i]))
            )


class TestObjectDetector(unittest.TestCase):
    # we have to define these here because mypy doesn't dive into the init hiding behind the classmethod
    object_detector = (
        NotImplemented
    )  # type: ClassVar[ZeroShotObjectDetectorWithFeedback]
    coke_bottle_image_bytes = NotImplemented  # type: ClassVar[bytes]
    default_labels = NotImplemented  # type: ClassVar[list[str]]
    default_incs = NotImplemented  # type: ClassVar[dict[str, list[str]]]
    default_excs = NotImplemented  # type: ClassVar[dict[str, list[str]]]
    default_nms_thre = NotImplemented  # type: ClassVar[float]
    default_conf_thres = NotImplemented  # type: ClassVar[dict[str, float]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.object_detector = ZeroShotObjectDetectorWithFeedback(jit=True)

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
        cls.default_nms_thre = 0.1
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

            self.assertEqual(len(batched_predicted_boxes), 1)
            predicted_boxes = batched_predicted_boxes[0]
            self.assertEqual(len(predicted_boxes), len(self.default_labels))
            bottle_boxes = predicted_boxes[0]
            moose_boxes = predicted_boxes[1]
            self.assertEqual(len(bottle_boxes), 9)
            self.assertEqual(len(moose_boxes), 0)

    @unittest.skip("We don't yet support batched object detection")
    def test_batched_detect(self) -> None:
        with torch.no_grad():
            random_images = torch.rand(16, 3, *self.object_detector.image_size)
            image_scale_ratios = torch.ones(16)

            single_image_outputs_list = []
            for image in random_images:
                single_image_outputs_list.append(
                    self.object_detector(
                        image.unsqueeze(0),
                        labels=self.default_labels,
                        inc_sub_labels_dict=self.default_incs,
                        exc_sub_labels_dict=None,
                        nms_thre=self.default_nms_thre,
                        label_conf_thres={"bottle": 0.0, "moose": 0.0},
                        image_scale_ratios=image_scale_ratios,
                    )[0]
                )

            batched_outputs = self.object_detector(
                random_images,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=None,
                nms_thre=self.default_nms_thre,
                label_conf_thres={"bottle": 0.0, "moose": 0.0},
                image_scale_ratios=image_scale_ratios,
            )

        for i in range(16):
            for j in range(2):
                from_batch = batched_outputs[i][j]
                from_single = single_image_outputs_list[i][j]
                self.assertEqual(from_batch.shape, from_single.shape)
                if from_batch.shape[0] == 0:
                    continue
                max_diff = (from_batch - from_single).abs().max().item()
                self.assertTrue(max_diff < 1e-5)
