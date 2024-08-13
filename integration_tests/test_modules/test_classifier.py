import unittest
import requests


class TestClassifier(unittest.TestCase):
    def test_classifier(self) -> None:
        response = requests.get("http://host.docker.internal:8000/classify")
        self.assertEqual(response.status_code, 200)
        self.assertIn("label", response.json())
        self.assertEqual(response.json()["label"], "cat")


