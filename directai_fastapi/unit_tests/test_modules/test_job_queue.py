import unittest
import asyncio
from multiprocessing import Queue, Process

from directai_fastapi.job_queue import JobQueue, worker_process, ExampleModel, ExampleModelConfig


class TestWorkerProcess(unittest.TestCase):
    def test_worker_process_exits(self) -> None:
        example_model_config = ExampleModelConfig(multiplier=2)
        input_queue = Queue()
        output_queue = Queue()
        
        worker_process = Process(target=worker_process, args=(0, example_model_config, input_queue, output_queue))
        worker_process.start()
        
        input_queue.put(None)
        
        worker_process.join()
        
        self.assertFalse(worker_process.is_alive())
        self.assertTrue(output_queue.empty())
        self.assertEqual(worker_process.exitcode, 0)
    
    def test_example_worker_process_multiplies(self) -> None:
        multiplier = 2
        example_model_config = ExampleModelConfig(multiplier=multiplier)
        input_queue = Queue()
        output_queue = Queue()
        
        worker_process = Process(target=worker_process, args=(0, example_model_config, input_queue, output_queue))
        worker_process.start()
        
        input_queue.put((0, 1))
        input_queue.put((1, 2))
        input_queue.put((2, 3))
        
        self.assertEqual(output_queue.get(), (0, 1 * multiplier))
        self.assertEqual(output_queue.get(), (1, 2 * multiplier))
        self.assertEqual(output_queue.get(), (2, 3 * multiplier))
        
        input_queue.put(None)
        
        worker_process.join()
        
        self.assertFalse(worker_process.is_alive())
        self.assertTrue(output_queue.empty())
        self.assertEqual(worker_process.exitcode, 0)
        
        