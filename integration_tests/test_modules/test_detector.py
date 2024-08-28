import os
import unittest
import requests
import numpy as np
from pydantic import BaseModel, Field, conlist
from typing import Dict, List, Union, Any
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

FASTAPI_HOST = "host.docker.internal"
FASTAPI_PORT = 8001


# copying over from directai_fastapi/pydantic_models.py
class SingleDetectionResponse(BaseModel):
    # see discussion: https://github.com/pydantic/pydantic/issues/975
    tlbr: conlist(float, min_items=4, max_items=4)  # type: ignore[valid-type]
    score: float
    class_: str = Field(alias="class")

    class Config:
        allow_population_by_field_name = True


def bbox_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    bb2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def compute_naive_bipartite_detection_loss(
    detections_set_1: List[SingleDetectionResponse],
    detections_set_2: List[SingleDetectionResponse],
    unmatched_loss: float = 1.0,
) -> float:
    """
    Computes the bipartite matching average loss between two sets of detections with potentially different sizes.
    If the classes are not equal, the loss is set to 1. Otherwise, the loss is the IoU of the bounding boxes.
    Unmatched detections are assigned a default loss.

    :param detections_set_1: List of detections (dicts with 'tlbr' and 'class' keys)
    :param detections_set_2: List of detections (dicts with 'tlbr' and 'class' keys)
    :param unmatched_loss: Default loss value for unmatched detections
    :return: Average loss as a float
    """
    num_detections_1 = len(detections_set_1)
    num_detections_2 = len(detections_set_2)
    cost_matrix = np.zeros((num_detections_1, num_detections_2))

    for i, det1 in enumerate(detections_set_1):
        for j, det2 in enumerate(detections_set_2):
            if det1.class_ != det2.class_:
                cost_matrix[i, j] = 1
            else:
                iou_score = bbox_iou(det1.tlbr, det2.tlbr)
                cost_matrix[i, j] = 1 - iou_score  # Loss is 1 - IoU

    # Apply the Hungarian algorithm to find the minimum cost matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Compute the total loss including unmatched detections
    total_loss = cost_matrix[row_ind, col_ind].sum()
    total_loss += unmatched_loss * (
        max(num_detections_1, num_detections_2) - len(row_ind)
    )  # Add loss for unmatched detections

    # Compute the average loss
    average_loss = total_loss / max(
        num_detections_1, num_detections_2
    )  # Normalizing by the larger set size
    return average_loss


class TestDetect(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName=methodName)
        self.endpoint = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/"

    def test_deploy_detector_config_missing(self) -> None:
        body: Dict[str, str] = {}
        expected_result = {
            "data": None,
            "message": "1 validation error for Request body -> detector_configs field required (type=value_error.missing)",
            "status_code": 422,
        }
        response = requests.post(self.endpoint + "deploy_detector", json=body)
        self.assertEqual(response.json(), expected_result)

    def test_deploy_detector_failure_include_missing(self) -> None:
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                }
            ],
            "nms_threshold": 0.4,
        }
        expected_result = {
            "status_code": 422,
            "message": "1 validation error for Request body -> detector_configs -> 0 -> examples_to_include field required (type=value_error.missing)",
            "data": None,
        }
        response = requests.post(self.endpoint + "deploy_detector", json=body)
        response_json = response.json()
        self.assertEqual(response_json, expected_result)

    def test_deploy_detector_success(self) -> None:
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                }
            ],
            "nms_threshold": 0.4,
        }
        response = requests.post(self.endpoint + "deploy_detector", json=body)
        response_json = response.json()
        self.assertTrue("deployed_id" in response_json)
        self.assertEqual(response_json["message"], "New model deployed.")

    def test_deploy_detector_success_without_exclude(self) -> None:
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "detection_threshold": 0.1,
                }
            ],
            "nms_threshold": 0.4,
        }
        response = requests.post(self.endpoint + "deploy_detector", json=body)
        response_json = response.json()
        self.assertTrue("deployed_id" in response_json)
        self.assertEqual(response_json["message"], "New model deployed.")

    def test_deploy_detector_success_without_detect_threshold(self) -> None:
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "examples_to_exclude": [],
                }
            ],
            "nms_threshold": 0.4,
        }
        response = requests.post(self.endpoint + "deploy_detector", json=body)
        response_json = response.json()
        self.assertTrue("deployed_id" in response_json)
        self.assertEqual(response_json["message"], "New model deployed.")

    def test_deploy_detector_success_without_nms_threshold(self) -> None:
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                }
            ],
        }
        response = requests.post(self.endpoint + "deploy_detector", json=body)
        response_json = response.json()
        self.assertTrue("deployed_id" in response_json)
        self.assertEqual(response_json["message"], "New model deployed.")

    def test_deploy_detector_success_without_augment_examples(self) -> None:
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                }
            ],
            "augment_examples": False,
        }
        response = requests.post(self.endpoint + "deploy_detector", json=body)
        response_json = response.json()
        self.assertTrue("deployed_id" in response_json)
        self.assertEqual(response_json["message"], "New model deployed.")


class TestDetectorInference(unittest.TestCase):
    def __init__(self, methodName: str = "runTest"):
        super().__init__(methodName=methodName)
        self.endpoint = f"http://{FASTAPI_HOST}:8000/"
        # here we assume that deploy has been tested and works
        # so we can generate a fixed deploy id for testing
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                }
            ],
        }
        deploy_response = requests.post(self.endpoint + "deploy_detector", json=body)
        deploy_response_json = deploy_response.json()
        self.sample_deployed_id = deploy_response_json["deployed_id"]

    def test_detect(self) -> None:
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        expected_detect_response_unaccelerated = [
            [
                {
                    "tlbr": [167.0, 151.0, 267.0, 494.0],
                    "score": 0.473,
                    "class": "bottle",
                },
                {
                    "tlbr": [407.0, 152.0, 510.0, 495.0],
                    "score": 0.466,
                    "class": "bottle",
                },
                {
                    "tlbr": [534.0, 150.0, 632.0, 494.0],
                    "score": 0.457,
                    "class": "bottle",
                },
                {"tlbr": [39.3, 183.0, 141.0, 494.0], "score": 0.45, "class": "bottle"},
                {
                    "tlbr": [653.0, 151.0, 756.0, 497.0],
                    "score": 0.45,
                    "class": "bottle",
                },
                {
                    "tlbr": [290.0, 160.0, 388.0, 493.0],
                    "score": 0.443,
                    "class": "bottle",
                },
                {
                    "tlbr": [780.0, 158.0, 879.0, 498.0],
                    "score": 0.418,
                    "class": "bottle",
                },
                {
                    "tlbr": [910.0, 108.0, 1030.0, 501.0],
                    "score": 0.385,
                    "class": "bottle",
                },
                {
                    "tlbr": [1070.0, 170.0, 1160.0, 498.0],
                    "score": 0.364,
                    "class": "bottle",
                },
                {
                    "tlbr": [27.3, 112.0, 1170.0, 507.0],
                    "score": 0.134,
                    "class": "bottle",
                },
            ]
        ]
        with open(sample_fp, "rb") as f:
            file_data = f.read()
        files = {
            "data": (sample_fp, file_data, "image/jpg"),
        }
        params = {"deployed_id": self.sample_deployed_id}
        response = requests.post(self.endpoint + "detect", params=params, files=files)
        detect_response_json = response.json()

        self.assertEqual(len(detect_response_json), 1)
        expected_detect_response = [
            SingleDetectionResponse.parse_obj(d)
            for d in expected_detect_response_unaccelerated[0]
        ]
        actual_detect_response = [
            SingleDetectionResponse.parse_obj(d) for d in detect_response_json[0]
        ]
        self.assertLess(
            compute_naive_bipartite_detection_loss(
                expected_detect_response, actual_detect_response
            ),
            0.05,
        )

    def test_detect_malformatted_image(self) -> None:
        sample_fp = "bad_file_path.suffix"
        file_data = b"This is not an image file"
        files = {
            "data": (sample_fp, file_data, "image/jpg"),
        }
        params = {"deployed_id": self.sample_deployed_id}
        response = requests.post(self.endpoint + "detect", params=params, files=files)
        response_json = response.json()
        self.assertEqual(response_json["status_code"], 422)
        self.assertEqual(
            response_json["message"], "Invalid image received, unable to open."
        )

    def test_detect_empty_image(self) -> None:
        sample_fp = "bad_file_path.jpg"
        files = {
            "data": (sample_fp, b"", "image/jpg"),
        }
        params = {"deployed_id": self.sample_deployed_id}
        response = requests.post(self.endpoint + "detect", params=params, files=files)
        response_json = response.json()
        self.assertEqual(response_json["status_code"], 422)
        self.assertEqual(
            response_json["message"], "Invalid image received, unable to open."
        )

    def test_detect_truncated_image(self) -> None:
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        with open(sample_fp, "rb") as f:
            file_data = f.read()
        file_data = file_data[: int(len(file_data) / 2)]
        files = {
            "data": (sample_fp, file_data, "image/jpg"),
        }
        params = {"deployed_id": self.sample_deployed_id}
        response = requests.post(self.endpoint + "detect", params=params, files=files)
        response_json = response.json()
        self.assertEqual(response_json["status_code"], 422)
        self.assertEqual(
            response_json["message"], "Invalid image received, unable to open."
        )

    def test_deploy_and_detect(self) -> None:
        # Starting Deploy Call
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.2,
                }
            ],
            "nms_threshold": 0.4,
        }
        deploy_response = requests.post(self.endpoint + "deploy_detector", json=body)
        deploy_response_json = deploy_response.json()
        self.assertTrue("deployed_id" in deploy_response_json)
        self.assertEqual(deploy_response_json["message"], "New model deployed.")
        deployed_id = deploy_response_json["deployed_id"]

        # Starting Detect Call
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        expected_detect_response_unaccelerated = [
            [
                {
                    "tlbr": [167.0, 151.0, 267.0, 494.0],
                    "score": 0.473,
                    "class": "bottle",
                },
                {
                    "tlbr": [407.0, 152.0, 510.0, 495.0],
                    "score": 0.466,
                    "class": "bottle",
                },
                {
                    "tlbr": [534.0, 150.0, 632.0, 494.0],
                    "score": 0.457,
                    "class": "bottle",
                },
                {"tlbr": [39.3, 183.0, 141.0, 494.0], "score": 0.45, "class": "bottle"},
                {
                    "tlbr": [653.0, 151.0, 756.0, 497.0],
                    "score": 0.45,
                    "class": "bottle",
                },
                {
                    "tlbr": [290.0, 160.0, 388.0, 493.0],
                    "score": 0.443,
                    "class": "bottle",
                },
                {
                    "tlbr": [780.0, 158.0, 879.0, 498.0],
                    "score": 0.418,
                    "class": "bottle",
                },
                {
                    "tlbr": [910.0, 108.0, 1030.0, 501.0],
                    "score": 0.385,
                    "class": "bottle",
                },
                {
                    "tlbr": [1070.0, 170.0, 1160.0, 498.0],
                    "score": 0.364,
                    "class": "bottle",
                },
            ]
        ]
        with open(sample_fp, "rb") as f:
            file_data = f.read()
        files = {
            "data": (sample_fp, file_data, "image/jpg"),
        }
        params = {"deployed_id": deployed_id}
        detect_response = requests.post(
            self.endpoint + "detect", params=params, files=files
        )
        detect_response_json = detect_response.json()

        self.assertEqual(len(detect_response_json), 1)
        expected_detect_response = [
            SingleDetectionResponse.parse_obj(d)
            for d in expected_detect_response_unaccelerated[0]
        ]
        actual_detect_response = [
            SingleDetectionResponse(**d) for d in detect_response_json[0]
        ]
        self.assertLess(
            compute_naive_bipartite_detection_loss(
                expected_detect_response, actual_detect_response
            ),
            0.05,
        )

    def test_deploy_with_long_prompt_and_detect(self) -> None:
        # Starting Deploy Call
        # NOTE: this is the only single-class detection test that runs the class-specific NMS algorithm
        very_long_prompt = "boat from birds-eye view maritime vessel from birds-eye view boat from top-down view maritime vessel from top-down view"
        body = {
            "detector_configs": [
                {
                    "name": "sample_prompt",
                    "examples_to_include": [very_long_prompt],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.01,
                }
            ],
            "nms_threshold": 0.4,
        }
        deploy_response = requests.post(self.endpoint + "deploy_detector", json=body)
        deploy_response_json = deploy_response.json()
        self.assertTrue("deployed_id" in deploy_response_json)
        self.assertEqual(deploy_response_json["message"], "New model deployed.")
        deployed_id = deploy_response_json["deployed_id"]

        # Starting Detect Call
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        expected_detect_response_unaccelerated = [
            [
                {
                    "tlbr": [2.04, -1.39, 1190.0, 636.0],
                    "score": 0.0195,
                    "class": "sample_prompt",
                }
            ]
        ]
        with open(sample_fp, "rb") as f:
            file_data = f.read()
        files = {
            "data": (sample_fp, file_data, "image/jpg"),
        }
        params = {"deployed_id": deployed_id}
        detect_response = requests.post(
            self.endpoint + "detect", params=params, files=files
        )
        detect_response_json = detect_response.json()

        self.assertEqual(len(detect_response_json), 1)
        expected_detect_response = [
            SingleDetectionResponse.parse_obj(d)
            for d in expected_detect_response_unaccelerated[0]
        ]
        actual_detect_response = [
            SingleDetectionResponse(**d) for d in detect_response_json[0]
        ]
        self.assertLess(
            compute_naive_bipartite_detection_loss(
                expected_detect_response, actual_detect_response
            ),
            0.05,
        )

    def test_deploy_and_detect_without_augmented_examples(self) -> None:
        # Starting Deploy Call
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.2,
                }
            ],
            "nms_threshold": 0.4,
            "augment_examples": False,
        }
        deploy_response = requests.post(self.endpoint + "deploy_detector", json=body)
        deploy_response_json = deploy_response.json()
        self.assertTrue("deployed_id" in deploy_response_json)
        self.assertEqual(deploy_response_json["message"], "New model deployed.")
        deployed_id = deploy_response_json["deployed_id"]

        # Starting Detect Call
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        expected_detect_response_accelerated = [
            [
                {
                    "tlbr": [
                        407.40325927734375,
                        152.23214721679688,
                        509.8586120605469,
                        494.7916564941406,
                    ],
                    "score": 0.6420959234237671,
                    "class": "bottle",
                },
                {
                    "tlbr": [
                        166.59225463867188,
                        151.0416717529297,
                        266.7410583496094,
                        493.6011962890625,
                    ],
                    "score": 0.6420406699180603,
                    "class": "bottle",
                },
                {
                    "tlbr": [
                        533.6681518554688,
                        150.29762268066406,
                        632.4032592773438,
                        494.3452453613281,
                    ],
                    "score": 0.6393750905990601,
                    "class": "bottle",
                },
                {
                    "tlbr": [
                        652.455322265625,
                        151.0416717529297,
                        755.8779907226562,
                        497.172607421875,
                    ],
                    "score": 0.6305534243583679,
                    "class": "bottle",
                },
                {
                    "tlbr": [
                        39.28571319580078,
                        182.5892791748047,
                        141.22023010253906,
                        494.1964111328125,
                    ],
                    "score": 0.6197347640991211,
                    "class": "bottle",
                },
                {
                    "tlbr": [
                        289.91815185546875,
                        160.26785278320312,
                        388.05804443359375,
                        492.7083435058594,
                    ],
                    "score": 0.5861819386482239,
                    "class": "bottle",
                },
                {
                    "tlbr": [
                        779.6502685546875,
                        157.5892791748047,
                        878.6830444335938,
                        497.7678527832031,
                    ],
                    "score": 0.5840385556221008,
                    "class": "bottle",
                },
                {
                    "tlbr": [
                        909.97021484375,
                        107.58928680419922,
                        1031.6964111328125,
                        501.33929443359375,
                    ],
                    "score": 0.5274810194969177,
                    "class": "bottle",
                },
                {
                    "tlbr": [
                        1065.4761962890625,
                        170.38690185546875,
                        1158.3333740234375,
                        497.4702453613281,
                    ],
                    "score": 0.49106982350349426,
                    "class": "bottle",
                },
            ]
        ]
        with open(sample_fp, "rb") as f:
            file_data = f.read()
        files = {
            "data": (sample_fp, file_data, "image/jpg"),
        }
        params = {"deployed_id": deployed_id}
        detect_response = requests.post(
            self.endpoint + "detect", params=params, files=files
        )
        detect_response_json = detect_response.json()

        self.assertEqual(len(detect_response_json), 1)
        expected_detect_response = [
            SingleDetectionResponse.parse_obj(d)
            for d in expected_detect_response_accelerated[0]
        ]
        actual_detect_response = [
            SingleDetectionResponse(**d) for d in detect_response_json[0]
        ]
        self.assertLess(
            compute_naive_bipartite_detection_loss(
                expected_detect_response, actual_detect_response
            ),
            0.05,
        )

        # now compare with the response from the same model with augmented examples
        body = {
            "detector_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": ["bottle"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.2,
                }
            ],
            "nms_threshold": 0.4,
            "augment_examples": True,
        }
        deploy_response = requests.post(self.endpoint + "deploy_detector", json=body)
        deploy_response_json = deploy_response.json()
        self.assertTrue("deployed_id" in deploy_response_json)
        self.assertEqual(deploy_response_json["message"], "New model deployed.")
        deployed_id_augmented_examples = deploy_response_json["deployed_id"]

        # Starting Detect Call
        params = {"deployed_id": deployed_id_augmented_examples}
        detect_response_augmented_examples = requests.post(
            self.endpoint + "detect", params=params, files=files
        )
        detect_response_augmented_examples_json = (
            detect_response_augmented_examples.json()
        )

        self.assertNotEqual(
            detect_response_json, detect_response_augmented_examples_json
        )

    def test_deploy_with_and_without_class_agnostic_nms(self) -> None:
        # Starting Deploy Call
        body = {
            "detector_configs": [
                {
                    "name": "face",
                    "examples_to_include": ["face"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                },
                {
                    "name": "head",
                    "examples_to_include": ["head"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                },
            ],
            "nms_threshold": 0.1,
            "class_agnostic_nms": True,
        }
        deploy_response = requests.post(self.endpoint + "deploy_detector", json=body)
        deploy_response_json = deploy_response.json()
        self.assertTrue("deployed_id" in deploy_response_json)
        self.assertEqual(deploy_response_json["message"], "New model deployed.")
        deployed_id = deploy_response_json["deployed_id"]

        # Starting Detect Call
        sample_fp = "sample_data/jumping_jack_up_isaac.jpg"
        with open(sample_fp, "rb") as f:
            file_data = f.read()
        files = {
            "data": (sample_fp, file_data, "image/jpg"),
        }
        params = {"deployed_id": deployed_id}
        detect_response = requests.post(
            self.endpoint + "detect", params=params, files=files
        )
        detect_response_json = detect_response.json()

        self.assertEqual(len(detect_response_json), 1)
        self.assertEqual(len(detect_response_json[0]), 1)
        detected_classes = set(
            [detection["class"] for detection in detect_response_json[0]]
        )
        self.assertEqual(detected_classes, {"head"})

        # now compare with the response from the same model without class agnostic nms
        # it should detect both a head and a face
        body = {
            "detector_configs": [
                {
                    "name": "face",
                    "examples_to_include": ["face"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                },
                {
                    "name": "head",
                    "examples_to_include": ["head"],
                    "examples_to_exclude": [],
                    "detection_threshold": 0.1,
                },
            ],
            "nms_threshold": 0.1,
            "class_agnostic_nms": False,
        }
        deploy_response = requests.post(self.endpoint + "deploy_detector", json=body)
        deploy_response_json = deploy_response.json()
        self.assertTrue("deployed_id" in deploy_response_json)
        self.assertEqual(deploy_response_json["message"], "New model deployed.")
        deployed_id_class_based_nms = deploy_response_json["deployed_id"]

        # Starting Detect Call
        params = {"deployed_id": deployed_id_class_based_nms}
        detect_response_class_based_nms = requests.post(
            self.endpoint + "detect", params=params, files=files
        )
        detect_response_class_based_nms_json = detect_response_class_based_nms.json()

        self.assertEqual(len(detect_response_class_based_nms_json), 1)
        self.assertEqual(len(detect_response_class_based_nms_json[0]), 2)
        detected_classes = set(
            [
                detection["class"]
                for detection in detect_response_class_based_nms_json[0]
            ]
        )
        self.assertEqual(detected_classes, {"head", "face"})
