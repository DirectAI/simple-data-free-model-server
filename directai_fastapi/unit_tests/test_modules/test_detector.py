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
    coke_can_image_bytes = NotImplemented  # type: ClassVar[bytes]
    default_labels = NotImplemented  # type: ClassVar[list[str]]
    default_incs = NotImplemented  # type: ClassVar[dict[str, list[str]]]
    default_excs = NotImplemented  # type: ClassVar[dict[str, list[str]]]
    default_nms_thre = NotImplemented  # type: ClassVar[float]
    default_conf_thres = NotImplemented  # type: ClassVar[dict[str, float]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.object_detector = ZeroShotObjectDetectorWithFeedback(jit=False)

        coke_bottle_filepath = "unit_tests/sample_data/coke_through_the_ages.jpeg"
        with open(coke_bottle_filepath, "rb") as f:
            cls.coke_bottle_image_bytes = f.read()
        coke_can_filepath = "unit_tests/sample_data/coke_can.jpg"
        with open(coke_can_filepath, "rb") as f:
            cls.coke_can_image_bytes = f.read()

        cls.default_labels = ["bottle", "can", "moose"]
        cls.default_incs = {
            "bottle": ["bottle", "glass bottle", "plastic bottle", "water bottle"],
            "can": ["can", "soda can", "aluminum can"],
            "moose": ["moose", "elk", "deer"],
        }
        cls.default_excs = {
            "bottle": ["can", "soda can", "aluminum can"],
        }
        cls.default_nms_thre = 0.1
        cls.default_conf_thres = {
            "bottle": 0.1,
            "can": 0.1,
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
            can_boxes = predicted_boxes[1]
            moose_boxes = predicted_boxes[2]
            self.assertEqual(len(bottle_boxes), 9)
            self.assertEqual(len(can_boxes), 0)
            self.assertEqual(len(moose_boxes), 0)

    def test_batched_detect(self) -> None:
        # ideally we would test a set of random images
        # but we use a sort, which has unstable ordering with floating point numbers
        # which means it is nontrivial to compare the outputs of the batched and single-image versions
        # instead we're going to limit to confident predictions from two images
        # and hope that the confidences are well-enough separated that the sort is stable

        with torch.no_grad():
            coke_bottle_image_tensor, coke_bottle_ratio = (
                created_padded_tensor_from_bytes(
                    self.coke_bottle_image_bytes, self.object_detector.image_size
                )
            )
            coke_can_image_tensor, coke_can_ratio = created_padded_tensor_from_bytes(
                self.coke_can_image_bytes, self.object_detector.image_size
            )

            batched_images = torch.cat(
                [
                    coke_bottle_image_tensor,
                ]
                * 8
                + [
                    coke_can_image_tensor,
                ]
                * 8,
                dim=0,
            )
            batched_ratios = torch.cat(
                [
                    coke_bottle_ratio,
                ]
                * 8
                + [
                    coke_can_ratio,
                ]
                * 8,
                dim=0,
            )

            single_image_outputs_list = []
            for image, ratio in zip(batched_images, batched_ratios):
                single_image_outputs_list.append(
                    self.object_detector(
                        image.unsqueeze(0),
                        labels=self.default_labels,
                        inc_sub_labels_dict=self.default_incs,
                        exc_sub_labels_dict=None,
                        nms_thre=self.default_nms_thre,
                        label_conf_thres=self.default_conf_thres,
                        image_scale_ratios=ratio.unsqueeze(0),
                    )[0]
                )

            batched_outputs = self.object_detector(
                batched_images,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=None,
                nms_thre=self.default_nms_thre,
                label_conf_thres=self.default_conf_thres,
                image_scale_ratios=batched_ratios,
            )

        for i in range(len(batched_outputs)):
            for j in range(len(batched_outputs[i])):
                from_batch = batched_outputs[i][j]
                from_single = single_image_outputs_list[i][j]
                self.assertEqual(from_batch.shape, from_single.shape)
                if from_batch.shape[0] == 0:
                    continue

                # these values have range on the order of 1e3, so we scale them to compare
                scale = torch.maximum(from_batch.abs(), from_single.abs())
                diff = (from_batch - from_single).abs()
                scaled_diff = diff / (scale + 1e-6)
                max_diff = scaled_diff.max().item()

                # large range and machine precision issues mean the max diff has a lot of noise
                # TODO: is this more than is sane?
                self.assertTrue(max_diff < 1e-3)

    def test_batch_detect_random_append(self) -> None:
        # we test batched detection by doing a pass for one image
        # and then doing a batched pass for that image and many random images
        # we use low confidence thresholds during the detection
        # and then truncate at a high confidence level to ensure stability due to sorting
        confidences = {label: 0.0 for label in self.default_labels}
        with torch.no_grad():
            coke_bottle_image_tensor, coke_bottle_ratio = (
                created_padded_tensor_from_bytes(
                    self.coke_bottle_image_bytes, self.object_detector.image_size
                )
            )
            baseline_output = self.object_detector(
                coke_bottle_image_tensor,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=None,
                nms_thre=self.default_nms_thre,
                label_conf_thres=confidences,
                image_scale_ratios=coke_bottle_ratio,
            )[0]

            random_tensors = torch.rand(128, 3, *self.object_detector.image_size)
            batched_tensor = torch.cat([random_tensors, coke_bottle_image_tensor])
            batched_ratios = torch.cat([torch.ones(128), coke_bottle_ratio])
            batched_output = self.object_detector(
                batched_tensor,
                labels=self.default_labels,
                inc_sub_labels_dict=self.default_incs,
                exc_sub_labels_dict=None,
                nms_thre=self.default_nms_thre,
                label_conf_thres=confidences,
                image_scale_ratios=batched_ratios,
            )[-1]

            for baseline_obj_detections, batched_obj_detections in zip(
                baseline_output, batched_output
            ):
                # filter by confidence of 0.1
                baseline_obj_detections = baseline_obj_detections[
                    baseline_obj_detections[:, 4] > 0.1
                ]
                batched_obj_detections = batched_obj_detections[
                    batched_obj_detections[:, 4] > 0.1
                ]

                self.assertEqual(
                    baseline_obj_detections.shape, batched_obj_detections.shape
                )
                if baseline_obj_detections.shape[0] == 0:
                    continue

                # these values have range on the order of 1e3, so we scale them to compare
                scale = torch.maximum(
                    baseline_obj_detections.abs(), batched_obj_detections.abs()
                )
                diff = (baseline_obj_detections - batched_obj_detections).abs()
                scaled_diff = diff / (scale + 1e-6)
                max_diff = scaled_diff.max().item()

                # large range and machine precision issues mean the max diff has a lot of noise
                self.assertTrue(max_diff < 1e-6)
