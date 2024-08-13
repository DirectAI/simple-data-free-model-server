import asyncio
from multiprocessing import Process, Queue
from pydantic import BaseModel
import torch
import time


class ExampleModelConfig(BaseModel):
    multiplier: int


class ExampleModel:
    def __init__(self, args: ExampleModelConfig):
        self.multiplier = args.multiplier
    
    def predict(self, x: int) -> int:
        # takes x*0.01 seconds to return x * multiplier
        time.sleep(x * 0.01)
        return x * self.multiplier


def start_worker_process_fn(gpu_index: int, model_args: ExampleModelConfig, input_queue: Queue, output_queue: Queue):
    model = ExampleModel(model_args)
    
    print(f"Worker {gpu_index} starting")
    
    while True:
        # NOTE: this could be modified to support batching
        # by taking the next N items from the input queue
        message = input_queue.get()
        if message is None:
            break
        task_id, x = message
        output_queue.put((task_id, model.predict(x)))
    
    print(f"Worker {gpu_index} exiting")


class JobQueue:
    def __init__(self, model_config: ExampleModelConfig, gpu_indices: list[int]|None = None):        
        self.model_config = model_config
        
        assert gpu_indices is not None, "gpu_indices must be provided for now"

        if gpu_indices is None:
            # use all available GPUs
            gpu_indices = list(range(torch.cuda.device_count()))
        
        self.result_queue = Queue()
        self.input_queues = []
        self.worker_processes = []
        self.next_task_id = 0
        self.pending_tasks = {}
        
        for gpu_index in gpu_indices:
            input_queue = Queue()
            self.input_queues.append(input_queue)
            worker_process = Process(target=start_worker_process_fn, args=(gpu_index, model_config, input_queue, self.result_queue))
            worker_process.start()
            self.worker_processes.append(worker_process)
        
        self.result_processing_loop_is_running = False
        
        print(f"Started {len(self.worker_processes)} worker processes")
    
    async def run(self):
        # self.loop.create_task(self._process_results())
        self.process_results_task = asyncio.create_task(self._process_results())
        self.result_processing_loop_is_running = True
    
    async def _process_results(self):
        while True:
            # multiprocessing doesn't play nicely with asyncio, so we need to poll the queue
            # print("Polling result queue")
            is_message = self.result_queue.empty()
            if not is_message:
                await asyncio.sleep(0.001)
                continue

            message = self.result_queue.get()
            if message is None:
                break
            
            task_id, result = message
            future = self.pending_tasks.pop(task_id)
            if future is not None:
                future.set_result(result)
    
    def predict(self, x: int) -> asyncio.Future:
        task_id = self.next_task_id
        self.next_task_id += 1
        
        future = asyncio.get_event_loop().create_future()
        self.pending_tasks[task_id] = future
        
        # choose input queue via round robin
        worker_id = task_id % len(self.input_queues)
        input_queue = self.input_queues[worker_id]
        input_queue.put((task_id, x))
        
        return future
    
    def check_workers_health(self) -> list[bool]:
        status = []
        for worker_process in self.worker_processes:
            status.append(worker_process.is_alive())
        return status
    
    async def close(self):
        for input_queue in self.input_queues:
            input_queue.put(None)
        
        for worker_process in self.worker_processes:
            worker_process.join()
        
        # signal the result processing loop to stop
        # NOTE: this will wait for all pending tasks to complete
        self.result_queue.put(None)
        await self.process_results_task
        self.result_processing_loop_is_running = False
        
        print("All worker processes exited")