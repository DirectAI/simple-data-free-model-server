import unittest
import requests


class TestDetector(unittest.TestCase):
    def test_classifier(self) -> None:
        response = requests.get("http://host.docker.internal:8000/detect")
        self.assertEqual(response.status_code, 200)
        self.assertIn("bboxes", response.json())
        self.assertEqual(response.json()["bboxes"], [[0.0, 0.0, 1.0, 1.0]])


