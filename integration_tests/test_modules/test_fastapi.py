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
        print(response.json())
        self.assertEqual(response.json(), expected_result)