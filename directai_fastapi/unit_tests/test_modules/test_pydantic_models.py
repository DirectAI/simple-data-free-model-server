import unittest
from pydantic_models import ClassifierDeploy, DetectorDeploy


class TestClassifierDeploy(unittest.TestCase):
    def test_classifier_deploy_from_config_dict(self) -> None:
        # Well-formatted config dict
        config_dict = {
            "labels": ["class1", "class2"],
            "inc_sub_labels_dict": {
                "class1": ["example1", "example2"],
                "class2": ["example3", "example4"],
            },
            "exc_sub_labels_dict": {
                "class1": ["example5"],
                "class2": ["example6"],
            },
            "augment_examples": True,
        }

        classifier_deploy = ClassifierDeploy.from_config_dict(config_dict)

        self.assertEqual(len(classifier_deploy.classifier_configs), 2)
        self.assertEqual(classifier_deploy.classifier_configs[0].name, "class1")
        self.assertEqual(
            classifier_deploy.classifier_configs[0].examples_to_include,
            ["example1", "example2"],
        )
        self.assertEqual(
            classifier_deploy.classifier_configs[0].examples_to_exclude, ["example5"]
        )
        self.assertEqual(classifier_deploy.classifier_configs[1].name, "class2")
        self.assertEqual(
            classifier_deploy.classifier_configs[1].examples_to_include,
            ["example3", "example4"],
        )
        self.assertEqual(
            classifier_deploy.classifier_configs[1].examples_to_exclude, ["example6"]
        )
        self.assertTrue(classifier_deploy.augment_examples)

    def test_classifier_deploy_from_config_dict_label_mismatch(self) -> None:
        # Config dict with an invalid label in inc_sub_labels_dict
        mismatch_config_dict = {
            "labels": ["class1", "class2"],
            "inc_sub_labels_dict": {
                "class1": ["example1", "example2"],
                "class3": ["example3", "example4"],  # "class3" is not in labels
            },
            "exc_sub_labels_dict": {
                "class1": ["example5"],
                "class2": ["example6"],
            },
            "augment_examples": True,
        }

        # with self.assertRaises(ValueError) as context:
        classifier_deploy = ClassifierDeploy.from_config_dict(mismatch_config_dict)
        self.assertEqual(len(classifier_deploy.classifier_configs), 2)
        self.assertEqual(classifier_deploy.classifier_configs[0].name, "class1")
        self.assertEqual(classifier_deploy.classifier_configs[1].name, "class3")
        self.assertNotIn(
            "class2", [config.name for config in classifier_deploy.classifier_configs]
        )


class TestDetectorDeploy(unittest.TestCase):
    def test_detector_deploy_from_config_dict(self) -> None:
        # Well-formatted config dict
        config_dict = {
            "labels": ["class1", "class2"],
            "inc_sub_labels_dict": {
                "class1": ["example1", "example2"],
                "class2": ["example3", "example4"],
            },
            "exc_sub_labels_dict": {
                "class1": ["example5"],
                "class2": ["example6"],
            },
            "label_conf_thres": {
                "class1": 0.2,
                "class2": 0.3,
            },
            "nms_threshold": 0.5,
            "class_agnostic_nms": False,
            "augment_examples": True,
            "deployed_id": "some_id",
        }

        detector_deploy = DetectorDeploy.from_config_dict(config_dict)

        self.assertEqual(len(detector_deploy.detector_configs), 2)
        self.assertEqual(detector_deploy.detector_configs[0].name, "class1")
        self.assertEqual(
            detector_deploy.detector_configs[0].examples_to_include,
            ["example1", "example2"],
        )
        self.assertEqual(
            detector_deploy.detector_configs[0].examples_to_exclude, ["example5"]
        )
        self.assertEqual(detector_deploy.detector_configs[0].detection_threshold, 0.2)
        self.assertEqual(detector_deploy.detector_configs[1].name, "class2")
        self.assertEqual(
            detector_deploy.detector_configs[1].examples_to_include,
            ["example3", "example4"],
        )
        self.assertEqual(
            detector_deploy.detector_configs[1].examples_to_exclude, ["example6"]
        )
        self.assertEqual(detector_deploy.detector_configs[1].detection_threshold, 0.3)
        self.assertEqual(detector_deploy.nms_threshold, 0.5)
        self.assertFalse(detector_deploy.class_agnostic_nms)
        self.assertTrue(detector_deploy.augment_examples)
        self.assertEqual(detector_deploy.deployed_id, "some_id")

    def test_detector_deploy_from_config_dict_label_mismatch(self) -> None:
        # Mismatched config dict
        mismatch_config_dict = {
            "labels": ["class1", "class3"],
            "inc_sub_labels_dict": {
                "class1": ["example1", "example2"],
                "class2": ["example3", "example4"],
            },
            "exc_sub_labels_dict": {
                "class1": ["example5"],
                "class2": ["example6"],
            },
            "label_conf_thres": {
                "class1": 0.2,
                "class2": 0.3,
            },
            "nms_threshold": 0.5,
            "class_agnostic_nms": False,
            "augment_examples": True,
            "deployed_id": "some_id",
        }

        detector_deploy = DetectorDeploy.from_config_dict(mismatch_config_dict)

        self.assertEqual(len(detector_deploy.detector_configs), 2)
        self.assertEqual(detector_deploy.detector_configs[0].name, "class1")
        self.assertEqual(detector_deploy.detector_configs[1].name, "class2")
        self.assertNotIn(
            "class3", [config.name for config in detector_deploy.detector_configs]
        )
