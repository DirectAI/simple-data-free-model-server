import os
import unittest
import requests

FASTAPI_HOST = "host.docker.internal"

class TestClassify(unittest.TestCase):
    def __init__(self, methodName: str ='runTest'):
        super().__init__(methodName=methodName)
        self.endpoint = f"http://{FASTAPI_HOST}:8000/"
    
    def test_deploy_classifier_config_missing(self) -> None:
        body: dict = {}
        expected_result = {
            'data': None,
            'message': '1 validation error for Request body -> classifier_configs field required (type=value_error.missing)',
            'status_code': 422
        }
        response = requests.post(
            self.endpoint+"deploy_classifier",
            json=body
        )
        self.assertEqual(response.json(), expected_result)
    
    def test_deploy_classifier_failure_include_missing(self) -> None:
        body = {
            "classifier_configs": [
                {
                    "name": "bottle",
                    "examples_to_exclude": []
                },
                {
                    "name": "can",
                    "examples_to_include": [
                        "can"
                    ],
                    "examples_to_exclude": []
                }
            ]
        }
        expected_result = {
            'status_code': 422,
            'message': '1 validation error for Request body -> classifier_configs -> 0 -> examples_to_include field required (type=value_error.missing)',
            'data': None
        }
        response = requests.post(
            self.endpoint+"deploy_classifier",
            json=body
        )
        response_json = response.json()
        self.assertEqual(response_json, expected_result)
    
    def test_deploy_classifier_success(self) -> None:
        body = {
            "classifier_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": [
                        "bottle"
                    ],
                    "examples_to_exclude": []
                },
                {
                    "name": "can",
                    "examples_to_include": [
                        "can"
                    ],
                    "examples_to_exclude": []
                }
            ]
        }
        response = requests.post(
            self.endpoint+"deploy_classifier",
            json=body
        )
        response_json = response.json()
        self.assertTrue('deployed_id' in response_json)
        self.assertEqual(response_json['message'], "New model deployed.")
    
    def test_deploy_classifier_success_without_exclude(self) -> None:
        body = {
            "classifier_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": [
                        "bottle"
                    ]
                },
                {
                    "name": "can",
                    "examples_to_include": [
                        "can"
                    ]
                }
            ]
        }
        response = requests.post(
            self.endpoint+"deploy_classifier",
            json=body
        )
        response_json = response.json()
        self.assertTrue('deployed_id' in response_json)
        self.assertEqual(response_json['message'], "New model deployed.")
    
    def test_deploy_classifier_success_without_augment_examples(self) -> None:
        body = {
            "classifier_configs": [
                {
                    "name": "bottle",
                    "examples_to_include": [
                        "bottle"
                    ]
                },
                {
                    "name": "can",
                    "examples_to_include": [
                        "can"
                    ]
                }
            ],
            "augment_examples": False
        }
        response = requests.post(
            self.endpoint+"deploy_classifier",
            json=body
        )
        response_json = response.json()
        self.assertTrue('deployed_id' in response_json)
        self.assertEqual(response_json['message'], "New model deployed.")