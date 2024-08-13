import unittest
import asyncio
from multiprocessing import Queue, Process
import time

from job_queue import JobQueue, start_worker_process_fn, ExampleModel, ExampleModelConfig


class TestWorkerProcess(unittest.TestCase):
    def test_worker_process_exits(self) -> None:
        example_model_config = ExampleModelConfig(multiplier=2)
        input_queue = Queue()
        output_queue = Queue()
        
        worker_process = Process(target=start_worker_process_fn, args=(0, example_model_config, input_queue, output_queue))
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
        
        worker_process = Process(target=start_worker_process_fn, args=(0, example_model_config, input_queue, output_queue))
        worker_process.start()
        
        start_time = time.time()
        
        input_queue.put((0, 1))
        input_queue.put((1, 2))
        input_queue.put((2, 3))
        
        self.assertEqual(output_queue.get(), (0, 1 * multiplier))
        self.assertEqual(output_queue.get(), (1, 2 * multiplier))
        self.assertEqual(output_queue.get(), (2, 3 * multiplier))
        
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # tasks take x*0.01 seconds to complete, so total elapsed time should be .01 + .02 + .03 = .06 seconds
        self.assertGreater(elapsed_time, .05)
        self.assertLess(elapsed_time, .07)
        
        input_queue.put(None)
        
        worker_process.join()
        
        self.assertFalse(worker_process.is_alive())
        self.assertTrue(output_queue.empty())
        self.assertEqual(worker_process.exitcode, 0)
        

class AsyncTestRunningJobQueue(unittest.IsolatedAsyncioTestCase):
    async def test_workers_spawn_and_close(self) -> None:
        example_model_config = ExampleModelConfig(multiplier=2)
        job_queue = JobQueue(model_config=example_model_config, gpu_indices=[0, 1, 2])

        self.assertEqual(len(job_queue.worker_processes), 3)
        self.assertTrue(all(job_queue.check_workers_health()))
        
        await job_queue.run()
        
        self.assertTrue(job_queue.result_processing_loop_is_running)

        await job_queue.close()

        self.assertFalse(any(job_queue.check_workers_health()))
        self.assertFalse(job_queue.result_processing_loop_is_running)


class AsyncTestJobQueueSubmitTasks(unittest.IsolatedAsyncioTestCase):
    async def test_job_queue_submit_task(self) -> None:
        multiplier = 2
        example_model_config = ExampleModelConfig(multiplier=multiplier)
        
        job_queue = JobQueue(model_config=example_model_config, gpu_indices=[0, 1, 2])
        await job_queue.run()
        
        awaitable = job_queue.predict(1)
        self.assertTrue(isinstance(awaitable, asyncio.Future))
        
        result = await awaitable
        
        self.assertEqual(result, 1 * multiplier)
        
        await job_queue.close()


# class AsyncTestJobQueueTaskTiming(unittest.IsolatedAsyncioTestCase):
#     async def test_job_queue_task_timing(self):
#         multiplier = 2
#         example_model_config = ExampleModelConfig(multiplier=multiplier)
        
#         gpu_indices = list(range(3))
#         job_queue = JobQueue(model_config=example_model_config, gpu_indices=gpu_indices)
        
#         await job_queue.run()
        
#         submition_values = list(range(9))
        
#         start_time = time.time()
        
#         awaitables = [job_queue.predict(x) for x in submition_values]
        
#         results = await asyncio.gather(*awaitables)
        
#         end_time = time.time()
        
#         elapsed_time = end_time - start_time
        
#         # tasks are submitted via round robin to the workers
#         # this means worker one should get tasks [0, 3, 6], worker two should get tasks [1, 4, 7], and worker three should get tasks [2, 5, 8]
#         # a task takes x*0.01 seconds to complete, so the total elapsed time should be the highest sum of x for each worker
#         # this is .02 + .05 + .08 = 1.5 seconds
#         self.assertGreater(elapsed_time, .14)
#         self.assertLess(elapsed_time, .16)
        
#         for i, x in enumerate(submition_values):
#             self.assertEqual(results[i], x * multiplier)
        
#         await job_queue.close()
        
