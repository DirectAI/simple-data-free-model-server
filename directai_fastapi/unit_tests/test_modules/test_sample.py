import unittest

class TestSample(unittest.TestCase):
    def __init__(self, *args: str, **kwargs: str) -> None:
        super().__init__(*args, **kwargs)

    def test_sample(self) -> None:
        self.assertTrue(True)