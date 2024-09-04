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
        mismatch_config_dict: dict = {
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
            "deployed_id": "sample_deployed_id",
        }

        with self.assertRaises(ValueError) as context:
            classifier_deploy = ClassifierDeploy.from_config_dict(mismatch_config_dict)

        self.assertIn(
            "Labels and inc_sub_labels_dict keys must be equal. exc_sub_labels_dict keys must be a subset of labels.",
            str(context.exception),
        )

        mismatch_config_dict["inc_sub_labels_dict"]["class2"] = mismatch_config_dict[
            "inc_sub_labels_dict"
        ]["class3"]
        del mismatch_config_dict["inc_sub_labels_dict"]["class3"]
        mismatch_config_dict["exc_sub_labels_dict"]["class3"] = ["example7"]

        with self.assertRaises(ValueError) as context2:
            classifier_deploy = ClassifierDeploy.from_config_dict(mismatch_config_dict)

        self.assertIn(
            "Labels and inc_sub_labels_dict keys must be equal. exc_sub_labels_dict keys must be a subset of labels.",
            str(context2.exception),
        )

    def test_valid_classifier_subset_exclusion(self) -> None:
        mismatch_config_dict = {
            "labels": ["class1", "class2"],
            "inc_sub_labels_dict": {
                "class1": ["example1", "example2"],
                "class2": ["example3", "example4"],
            },
            "exc_sub_labels_dict": {
                "class1": ["example5"],
            },
            "augment_examples": True,
            "deployed_id": "sample_deployed_id",
        }

        classifier_deploy = ClassifierDeploy.from_config_dict(mismatch_config_dict)
        self.assertEqual(
            classifier_deploy.classifier_configs[1].examples_to_exclude, []
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
        mismatch_config_dict: dict = {
            "labels": ["class1", "class2"],
            "inc_sub_labels_dict": {
                "class1": ["example1", "example2"],
                "class3": ["example3", "example4"],
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

        with self.assertRaises(ValueError) as context:
            detector_deploy = DetectorDeploy.from_config_dict(mismatch_config_dict)

        self.assertIn(
            "Labels and inc_sub_labels_dict keys must be equal. exc_sub_labels_dict keys must be a subset of labels.",
            str(context.exception),
        )

        mismatch_config_dict["inc_sub_labels_dict"]["class2"] = mismatch_config_dict[
            "inc_sub_labels_dict"
        ]["class3"]
        del mismatch_config_dict["inc_sub_labels_dict"]["class3"]
        mismatch_config_dict["exc_sub_labels_dict"]["class3"] = ["example7"]

        with self.assertRaises(ValueError) as context:
            detector_deploy = DetectorDeploy.from_config_dict(mismatch_config_dict)

        self.assertIn(
            "Labels and inc_sub_labels_dict keys must be equal. exc_sub_labels_dict keys must be a subset of labels.",
            str(context.exception),
        )

    def test_valid_detector_subset_exclusion(self) -> None:
        mismatch_config_dict = {
            "labels": ["class1", "class2"],
            "inc_sub_labels_dict": {
                "class1": ["example1", "example2"],
                "class2": ["example3", "example4"],
            },
            "exc_sub_labels_dict": {
                "class1": ["example5"],
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
        self.assertEqual(detector_deploy.detector_configs[1].examples_to_exclude, [])
