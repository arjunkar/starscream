"""
Parallel optimizer for Starscream interacting with data parallelism
and pipeline parallelism.
"""


import torch
import parallel_state as ps
from distributed import DistributedDataParallel
from schedules import Schedule


class ParallelSGDOptimizer():
    def __init__(self,
                 model: DistributedDataParallel,
                 sched: Schedule,
                 lr=5e-3) -> None:
        assert ps.model_parallel_is_initialized(), \
            "Parallel state is not initialized."
        # sched must have model as its model
        self.model = model
        self.sched = sched
        self.lr = lr

    def zero_grad(self):
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param.data)

    def step(self):
        # allreduce_gradients must be called before this
        # update parameters
        for param in self.model.parameters():
            param.data -= self.lr * param.grad
        # flush pipeline
        self.sched.input_store = {}
        self.sched.output_store = {}
        self.sched.send_reqs = {}
        self.sched.recv_reqs = {}
        # load new data
        self.sched.init_comm()