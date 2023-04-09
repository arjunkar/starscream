"""
Pipeline schedules for Starscream.

Loosely inspired by Megatron:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import parallel_state as ps
from collections import deque

# Types
from abc import ABC, abstractmethod
from typing import Union
from distributed import DistributedDataParallel
ModelType = Union[DistributedDataParallel, nn.Module]


class Schedule(ABC):
    def __init__(self, 
                 model: ModelType,
                 data_loader,
                 loss_fn,
                 num_micro_batches: int
                 ) -> None:

        assert ps.model_parallel_is_initialized(), \
            "Parallel state is not initialized."
        assert ps.get_pipeline_model_parallel_world_size() > 1, \
            "Pipeline model parallel size must be at least 2 to use schedules."

        self.is_first = ps.is_pipeline_first_stage()
        self.is_last = ps.is_pipeline_last_stage()
        self.first = ps.get_pipeline_model_parallel_first_rank()
        self.last = ps.get_pipeline_model_parallel_last_rank()
        self.next = ps.get_pipeline_model_parallel_next_rank()
        self.prev = ps.get_pipeline_model_parallel_prev_rank()

        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.num_micro_batches = num_micro_batches

        self.input_store = {}
        self.output_store = {}
        self.send_reqs = {}
        self.recv_reqs = {}

        # First pipeline stage sends classes to last
        if self.is_first:
            for num in range(1, num_micro_batches+1):
                self.input_store[num], ans = self.data_loader.__next__()
                shape = torch.tensor(list(ans.size()))
                dist.send(
                    tensor=shape,
                    dst=self.last,
                    tag=0
                )
                # self.output_store[-num] = ans.detach()
                self.send_reqs[-num] = dist.isend(
                    tensor=ans,
                    dst=self.last,
                    tag=num)
                self.send_reqs[-num].wait()
            
        # Last pipeline stage receives classes from first
        if self.is_last:
            for num in range(1, num_micro_batches+1):
                # answer shape should be a vector of classes
                shape = torch.tensor([0])
                dist.recv(
                    tensor=shape,
                    src=self.first,
                    tag=0
                )
                self.input_store[-num] = torch.empty(size=shape.tolist(),
                                                     dtype=torch.int64)
                self.recv_reqs[-num] = dist.irecv(
                    tensor=self.input_store[-num],
                    src=self.first,
                    tag=num
                )
                self.recv_reqs[-num].wait()

    def fwd_step(self, num: int):
        # Incoming communication handling
        if not self.is_first:
            shape = torch.tensor([0, 0])
            shape_req = dist.irecv(
                tensor=shape,
                src=self.prev,
                tag=0
            )
            shape_req.wait()
            self.input_store[num] = torch.empty(size=shape.tolist())
            # Receive input from previous stage
            self.recv_reqs[num] = dist.irecv(
                tensor=self.input_store[num],
                src=self.prev,
                tag=num
            )
            self.recv_reqs[num].wait()

        # Forward pass
        input = self.input_store[num].requires_grad_(True)
        self.output_store[num] = self.model(input)
        
        # Outgoing communication handling
        if not self.is_last:
            # Pre-allocate for backward step
            self.input_store[-num] = torch.empty_like(self.output_store[num])
            shape_req = dist.isend(
                tensor=torch.tensor(
                    list(
                        self.output_store[num].size()
                        )),
                dst=self.next,
                tag=0
            )
            shape_req.wait()
            # Send outputs to next stage
            self.send_reqs[num] = dist.isend(
                tensor=self.output_store[num],
                dst=self.next,
                tag=num
            )
            self.send_reqs[num].wait()

    def bwd_step(self, num: int):
        # Incoming communication handling
        if self.is_last:
            ans = self.input_store[-num]
            out = self.loss_fn(self.output_store[num], ans)
            out_grads = None
        else:
            # Receive gradients from next stage
            self.recv_reqs[-num] = dist.irecv(
                # Pre-allocated during forward step
                tensor=self.input_store[-num],
                src=self.next,
                tag=num
            )
            # Ensure downstream operations in pipeline
            # are all completed
            self.recv_reqs[-num].wait()
            out = self.output_store[num]
            out_grads = self.input_store[-num]
        
        # Backward pass
        out.backward(gradient=out_grads)

        # Outgoing communication handling
        if not self.is_first:
            self.output_store[-num] = self.input_store[num].grad.detach()
            self.send_reqs[-num] = dist.isend(
                tensor=self.output_store[-num],
                dst=self.prev,
                tag=num
            )
            self.send_reqs[-num].wait()

    @abstractmethod
    def train_step(self):
        pass


class GPipe(Schedule):
    def __init__(self, 
                 model: ModelType,
                 data_loader,
                 loss_fn,
                 num_micro_batches: int
                 ) -> None:
        super().__init__(model, data_loader, loss_fn, num_micro_batches)

    def train_step(self):
        fwd_schedule = torch.arange(self.num_micro_batches) + 1
        bwd_schedule = -torch.arange(self.num_micro_batches) - 1
        queue = deque(
            fwd_schedule.tolist() + bwd_schedule.tolist()
        )
        # Perform parallel pipeline work
        while queue:
            job = queue.popleft()
            if job > 0:
                self.fwd_step(job)
            else:
                self.bwd_step(-job)


class PipeDream(Schedule):
    pass # Implement PipeDream