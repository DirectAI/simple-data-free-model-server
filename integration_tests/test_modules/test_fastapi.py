import os
import unittest
import requests
import numpy as np
from typing import Dict, Any

FASTAPI_HOST = "host.docker.internal"

def compute_kl_divergence_based_classification_loss(response1: Dict[str, Any] , response2: Dict[str, Any]) -> float:
    """
    Computes the Kullback-Leibler divergence between two sets of predicted probabilities.

    :param response1: First dictionary containing 'pred', 'raw_scores', and 'scores'
    :param response2: Second dictionary containing 'pred', 'raw_scores', and 'scores'
    :return: KL divergence as a float
    """
    # Extract the predicted scores from both responses
    scores1 = np.array(list(response1['scores'].values()))
    scores2 = np.array(list(response2['scores'].values()))

    # Compute the KL divergence
    kl_divergence = np.sum(scores1 * np.log(scores1 / scores2))
    return kl_divergence

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
    
    @unittest.skip("classifier isn't built yet")
    def test_classify(self) -> None:
        sample_deployed_id = "b5d1bdec-3bf3-4632-a011-538384d5bcb1"
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        with open(sample_fp, 'rb') as f:
            file_data = f.read()
        files = {
            'data': (sample_fp, file_data, "image/jpg"),
        }
        params = {
            'deployed_id': sample_deployed_id
        }
        expected_response = {
            'scores': {
                'bottle': 0.6345123648643494,
                'can': 0.36548763513565063
            },
            'ed': 'bottle',
            'raw_scores': {
                'bottle': 0.6409257650375366,
                'can': 0.6023120284080505
            }
        }
        response = requests.post(
            self.endpoint+"classify",
            params=params,
            files=files
        )
        self.assertEqual(response.status_code, 200)
        response_json = response.json()
        
        self.assertLess(
            compute_kl_divergence_based_classification_loss(
                expected_response,
                response_json
            ),
            2e-5
        )
    
    def test_classify_malformatted_image(self) -> None:
        sample_deployed_id = "b5d1bdec-3bf3-4632-a011-538384d5bcb1"
        sample_fp = "bad_file_path.suffix"
        file_data =  b"This is not an image file"
        files = {
            'data': (sample_fp, file_data, "image/jpg"),
        }
        params = {
            'deployed_id': sample_deployed_id
        }
        response = requests.post(
            self.endpoint+"classify",
            params=params,
            files=files
        )
        response_json = response.json()
        self.assertEqual(response_json['status_code'],422)
        self.assertEqual(
            response_json['message'],
            "Invalid image received, unable to open."
        )
    
    def test_classify_empty_image(self) -> None:
        sample_deployed_id = "b5d1bdec-3bf3-4632-a011-538384d5bcb1"
        sample_fp = "bad_file_path.suffix"
        file_data =  b""
        files = {
            'data': (sample_fp, file_data, "image/jpg"),
        }
        params = {
            'deployed_id': sample_deployed_id
        }
        response = requests.post(
            self.endpoint+"classify",
            params=params,
            files=files
        )
        response_json = response.json()
        self.assertEqual(response_json['status_code'],422)
        self.assertEqual(
            response_json['message'],
            "Invalid image received, unable to open."
        )
    
    def test_classify_truncated_image(self) -> None:
        sample_deployed_id = "b5d1bdec-3bf3-4632-a011-538384d5bcb1"
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        with open(sample_fp, 'rb') as f:
            file_data = f.read()
        file_data = file_data[:int(len(file_data)/2)]
        files = {
            'data': (sample_fp, file_data, "image/jpg"),
        }
        params = {
            'deployed_id': sample_deployed_id
        }
        response = requests.post(
            self.endpoint+"classify",
            params=params,
            files=files
        )
        response_json = response.json()
        self.assertEqual(response_json['status_code'],422)
        self.assertEqual(
            response_json['message'],
            "Invalid image received, unable to open."
        )
    
    @unittest.skip("classifier isn't built yet")
    def test_deploy_and_classify(self) -> None:
        # Starting Deploy Call
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
        deploy_response = requests.post(
            self.endpoint+"deploy_classifier",
            json=body
        )
        deploy_response_json = deploy_response.json()
        self.assertTrue('deployed_id' in deploy_response_json)
        self.assertEqual(deploy_response_json['message'], "New model deployed.")
        deployed_id = deploy_response_json['deployed_id']
        
        # Starting Classify Call
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        with open(sample_fp, 'rb') as f:
            file_data = f.read()
        files = {
            'data': (sample_fp, file_data, "image/jpg"),
        }
        params = {
            'deployed_id': deployed_id
        }
        expected_response = {
            'scores': {
                'bottle': 0.6345123648643494,
                'can': 0.36548763513565063
            },
            'pred': 'bottle',
            'raw_scores': {
                'bottle': 0.6409257650375366,
                'can': 0.6023120284080505
            }
        }
        classify_response = requests.post(
            self.endpoint+"classify",
            params=params,
            files=files
        )
        self.assertEqual(classify_response.status_code, 200)
        classify_response_json = classify_response.json()
                
        self.assertLess(
            compute_kl_divergence_based_classification_loss(
                expected_response,
                classify_response_json
            ),
            2e-5
        )
    
    @unittest.skip("classifier isn't built yet")
    def test_deploy_and_classify_without_augment_examples(self) -> None:        
        # deploy without augment_examples
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
            ],
            "augment_examples": False
        }
        deploy_response = requests.post(
            self.endpoint+"deploy_classifier",
            json=body
        )
        deploy_response_json = deploy_response.json()
        self.assertTrue('deployed_id' in deploy_response_json)
        self.assertEqual(deploy_response_json['message'], "New model deployed.")
        deployed_id = deploy_response_json['deployed_id']
        
        # Starting Classify Call
        sample_fp = "sample_data/coke_through_the_ages.jpeg"
        with open(sample_fp, 'rb') as f:
            file_data = f.read()
        files = {
            'data': (sample_fp, file_data, "image/jpg"),
        }
        params = {
            'deployed_id': deployed_id
        }
        expected_response = {
            'scores': {
                'bottle': 0.5853425860404968,
                'can': 0.4146574139595032
            },
            'pred': 'bottle',
            'raw_scores': {
                'bottle': 0.6374971270561218,
                'can': 0.6133649945259094
            }
        }
        classify_response = requests.post(
            self.endpoint+"classify",
            params=params,
            files=files
        )
        self.assertEqual(classify_response.status_code, 200)
        classify_response_json = classify_response.json()
                
        self.assertLess(
            compute_kl_divergence_based_classification_loss(
                expected_response,
                classify_response_json
            ),
            2e-5
        )
        
        # now compare with the response from the deploy and classify with augment_examples
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
            ],
            "augment_examples": True
        }
        deploy_response = requests.post(
            self.endpoint+"deploy_classifier",
            json=body
        )
        deploy_response_json = deploy_response.json()
        self.assertTrue('deployed_id' in deploy_response_json)
        self.assertEqual(deploy_response_json['message'], "New model deployed.")
        deployed_id_with_augmented_examples = deploy_response_json['deployed_id']
        
        # Starting Classify Call
        params = {
            'deployed_id': deployed_id_with_augmented_examples
        }
        classify_response_augmented = requests.post(
            self.endpoint+"classify",
            params=params,
            files=files
        )
        self.assertEqual(classify_response_augmented.status_code, 200)
        classify_response_augmented_json = classify_response_augmented.json()
        
        self.assertNotEqual(classify_response_json, classify_response_augmented_json)