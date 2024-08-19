import unittest
from logging_config import (
    logging_level_from_str, 
    grab_importing_fp
)

class TestLoggingLevelFromStr(unittest.TestCase):
    def test_info_level_returns(self) -> None:
        correct_answers = {
            "NOTSET": 0,
            "DEBUG": 10,
            "INFO": 20,
            "": 20,
            "WARNING": 30,
            "ERROR": 40,
            "CRITICAL": 50
        }
        for k in correct_answers:
            ans = correct_answers[k]
            self.assertEqual(
                logging_level_from_str(k),
                ans
            )

class TestFileImport(unittest.TestCase):
    def test_file_import_fp(self) -> None:
        full_path = grab_importing_fp()
        self.assertTrue(full_path.endswith('unittest/__main__.py'))
