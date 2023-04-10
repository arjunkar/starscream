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

        self.local_rank = ps.get_pipeline_model_parallel_rank()
        self.group = ps.get_pipeline_model_parallel_group()
        self.world_size = ps.get_pipeline_model_parallel_world_size()

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

        self.init_comm()

    def init_comm(self):
        # First pipeline stage sends classes to last
        if self.is_first:
            for num in range(1, self.num_micro_batches+1):
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
            for num in range(1, self.num_micro_batches+1):
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
        # print("Process ", dist.get_rank(), "incoming fwd", num)
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
        # print("Process ", dist.get_rank(), "forwarding ", num)
        input = self.input_store[num].requires_grad_(True)
        self.output_store[num] = self.model(input)
        
        # Outgoing communication handling
        # print("Process ", dist.get_rank(), "outgoing fwd", num)
        if not self.is_last:
            # Pre-allocate for backward step
            self.input_store[-num] = torch.empty_like(self.output_store[num])
            # Buffer irecv request.
            # The placement of this irecv in the forward pass rather than
            # at the start of the backward pass is critical to prevent deadlock
            # in the 1F1B pipeline.
            # Without it, the second forward pass at the penultimate stage can deadlock
            # with the first backward pass of the final stage, as both attempt
            # to wait() on isend requests without any irecv requests in the 
            # communication queue.
            # Placing it here ensures that there is an available irecv for the
            # final stage to match and continue processing.
            self.recv_reqs[-num] = dist.irecv(
                tensor=self.input_store[-num],
                src=self.next,
                tag=num
            )
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
        # print("Process ", dist.get_rank(), "incoming bwd", num)
        if self.is_last:
            ans = self.input_store[-num]
            micro_batch_size = ans.size()[0]
            # Square the number of microbatches in denominator
            out = self.loss_fn(self.output_store[num], ans) / (self.num_micro_batches**2 * micro_batch_size)
            out_grads = None
        else:
            # Receive gradients from next stage.
            # Ensures downstream operations in pipeline
            # are all completed.
            self.recv_reqs[-num].wait()
            out = self.output_store[num]
            out_grads = self.input_store[-num]
        
        # Backward pass
        # ("Process ", dist.get_rank(), "backwarding ", num)
        out.backward(gradient=out_grads)

        # Outgoing communication handling
        # print("Process ", dist.get_rank(), "outgoing bwd ", num)
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
        # See Figure 3 in arXiv:2104.04473
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
        # Ensure all processes have finished working
        dist.barrier()


class PipeDream(Schedule):
    def __init__(self, 
                 model: ModelType,
                 data_loader,
                 loss_fn,
                 num_micro_batches: int
                 ) -> None:
        super().__init__(model, data_loader, loss_fn, num_micro_batches)

        assert self.world_size < self.num_micro_batches, \
            "PipeDream requires pipeline parallel size < number of microbatches."

    def train_step(self):
        num_mb = self.num_micro_batches
        rank = self.local_rank
        world = self.world_size
        # See Figure 4 in arXiv:2104.04473
        if self.is_last:
            fwds = list(range(1, num_mb+1))
            bwds = list(range(-1, -num_mb-1, -1))
            sched = fwds + bwds
            sched[::2] = fwds
            sched[1::2] = bwds
            queue = deque(sched)
        else:
            warmup_schedule = [i for i in range(1, world+1)] + [-j for j in range(1, rank+2)]
            steady_fwds = [i for i in range(world+1, num_mb+1)]
            steady_bwds = [-(val-(world-1)+rank) for val in steady_fwds]
            steady_schedule = steady_fwds + steady_bwds
            # Steady state of 1 forward, 1 backward
            steady_schedule[::2] = steady_fwds
            steady_schedule[1::2] = steady_bwds
            cooldown_schedule = [-j for j in range(-steady_bwds[-1]+1, num_mb+1)]
            queue = deque(
                warmup_schedule + steady_schedule + cooldown_schedule
            )
        # Perform parallel pipeline work
        while queue:
            job = queue.popleft()
            if job > 0:
                self.fwd_step(job)
            else:
                self.bwd_step(-job)
        # Ensure all processes have finished working
        dist.barrier()


class Interleaved(Schedule):
    def __init__(self, 
                 model: ModelType,
                 data_loader,
                 loss_fn,
                 num_micro_batches: int
                 ) -> None:
        super().__init__(model, data_loader, loss_fn, num_micro_batches)

    def train_step(self):
        pass # Implement interleaved 1F1B