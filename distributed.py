"""
Simple version of PyTorch's DistributedDataParallel to manage
data parallelism in Starscream.

Inspired by Megatron's LocalDDP:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/distributed.py
"""

from torch import nn
import torch.distributed as dist
import parallel_state as ps


class DistributedDataParallel(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def broadcast_params(self):
        group = ps.get_data_parallel_group()
        src_rank = ps.get_data_parallel_src_rank()
        for param in self.module.parameters():
            dist.broadcast(tensor=param.data,
                           src=src_rank,
                           group=group)
            
    def allreduce_gradients(self):
        group = ps.get_data_parallel_group()
        world_size = ps.get_data_parallel_world_size()
        for param in self.module.parameters():
            dist.all_reduce(tensor=param.grad.data,
                            op=dist.ReduceOp.SUM,
                            group=group)
            param.grad.data /= world_size