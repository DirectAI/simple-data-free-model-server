import asyncio
from multiprocessing import Process, Queue
from pydantic import BaseModel
import torch


class ExampleModelConfig(BaseModel):
    multiplier: int


class ExampleModel:
    def __init__(self, args: ExampleModelConfig):
        self.multiplier = args.multiplier
    
    def predict(self, x: int) -> int:
        return x * self.multiplier


def worker_process(gpu_index: int, model_args: ExampleModelConfig, input_queue: Queue, output_queue: Queue):
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
        self.loop = asyncio.get_event_loop()
        
        self.model_config = model_config
        
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
            worker_process = Process(target=worker_process, args=(gpu_index, model_config, input_queue, self.result_queue))
            worker_process.start()
            self.worker_processes.append(worker_process)
        
        print(f"Started {len(self.worker_processes)} worker processes")
    
        self.loop.create_task(self._process_results())
    
    async def _process_results(self):
        while True:
            message = await self.loop.run_in_executor(None, self.result_queue.get)
            if message is None:
                break
            
            task_id, result = message
            future = self.pending_tasks.pop(task_id)
            if future is not None:
                future.set_result(result)
    
    def predict(self, x: int) -> asyncio.Future:
        task_id = self.next_task_id
        self.next_task_id += 1
        
        future = self.loop.create_future()
        self.pending_tasks[task_id] = future
        
        # choose the input queue with the shortest length
        input_queue = min(self.input_queues, key=lambda q: q.qsize())
        input_queue.put((task_id, x))
        
        return future
    
    def close(self):
        for input_queue in self.input_queues:
            input_queue.put(None)
        
        for worker_process in self.worker_processes:
            worker_process.join()
        
        print("All worker processes exited")