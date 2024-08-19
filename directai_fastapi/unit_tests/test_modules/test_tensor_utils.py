import unittest
import torch

from modeling.tensor_utils import (
    image_bytes_to_tensor,
    batch_encode_cache_missed_list_elements,
)
from lru import LRU


class TestImageBytesToTensor(unittest.TestCase):
    def test_image_bytes_to_tensor(self) -> None:
        sample_filepath = "unit_tests/sample_data/coke_through_the_ages.jpeg"
        with open(sample_filepath, "rb") as f:
            image_bytes = f.read()

        image_size = (224, 224)
        tensor = image_bytes_to_tensor(image_bytes, image_size)

        self.assertEqual(tensor.shape, (1, 3, 224, 224))


class TestBatchEncodeCacheMissedListElements(unittest.TestCase):
    def test_batch_encode_cache_missed_list_elements(self) -> None:
        n_args_passed_to_encode_fn = 0

        def encode_fn(args_list: list[str]) -> torch.Tensor:
            nonlocal n_args_passed_to_encode_fn
            n_args_passed_to_encode_fn += len(args_list)
            return torch.tensor([len(arg) for arg in args_list])

        args_list = ["a", "bb", "ccc", "dddd"]
        cache: dict[str, torch.Tensor] = {}

        output_tensor = batch_encode_cache_missed_list_elements(
            encode_fn, args_list, cache
        )
        self.assertTrue(torch.equal(output_tensor, torch.tensor([1, 2, 3, 4])))
        self.assertEqual(n_args_passed_to_encode_fn, 4)
        for arg in args_list:
            self.assertIn(arg, cache)
            self.assertEqual(cache[arg], len(arg))

        new_args_list = ["bb", "ccc", "dddd", "eeeee", "ffffff"]
        new_output_tensor = batch_encode_cache_missed_list_elements(
            encode_fn, new_args_list, cache
        )
        self.assertTrue(torch.equal(new_output_tensor, torch.tensor([2, 3, 4, 5, 6])))
        self.assertEqual(n_args_passed_to_encode_fn, 6)

    def test_lru_cache(self) -> None:
        n_args_passed_to_encode_fn = 0

        def encode_fn(args_list: list[str]) -> torch.Tensor:
            nonlocal n_args_passed_to_encode_fn
            n_args_passed_to_encode_fn += len(args_list)
            return torch.tensor([len(arg) for arg in args_list])

        args_list = ["a", "bb", "ccc", "dddd"]
        cache: LRU = LRU(2)

        output_tensor = batch_encode_cache_missed_list_elements(
            encode_fn, args_list, cache
        )
        assert isinstance(output_tensor, torch.Tensor)
        self.assertTrue(torch.equal(output_tensor, torch.tensor([1, 2, 3, 4])))
        self.assertEqual(n_args_passed_to_encode_fn, 4)
        self.assertNotIn("a", cache)
        self.assertNotIn("bb", cache)
        self.assertIn("ccc", cache)
        self.assertIn("dddd", cache)

        new_args_list = ["a", "ccc"]
        new_output_tensor = batch_encode_cache_missed_list_elements(
            encode_fn, new_args_list, cache
        )
        self.assertTrue(torch.equal(new_output_tensor, torch.tensor([1, 3])))
        self.assertEqual(n_args_passed_to_encode_fn, 5)
        self.assertIn("a", cache)
        self.assertNotIn("bb", cache)
        self.assertIn("ccc", cache)
        self.assertNotIn("dddd", cache)

    def test_max_cache_size_0(self) -> None:
        n_args_passed_to_encode_fn = 0

        def encode_fn(args_list: list[str]) -> torch.Tensor:
            nonlocal n_args_passed_to_encode_fn
            n_args_passed_to_encode_fn += len(args_list)
            return torch.tensor([len(arg) for arg in args_list])

        args_list = ["a", "bb", "ccc", "dddd"]
        cache = None

        _ = batch_encode_cache_missed_list_elements(encode_fn, args_list, cache)
        self.assertEqual(n_args_passed_to_encode_fn, 4)
        _ = batch_encode_cache_missed_list_elements(encode_fn, args_list, cache)
        self.assertEqual(n_args_passed_to_encode_fn, 8)
