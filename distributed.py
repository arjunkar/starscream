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

        assert ps.model_parallel_is_initialized(), \
            "Parallel state is not initialized."
        assert ps.get_data_parallel_world_size() > 1, \
            "Data parallel size must be at least 2 to use DataDistributedParallel."
        
        self.module = module
        self.src_rank = ps.get_data_parallel_src_rank()
        self.group = ps.get_data_parallel_group()
        self.world_size = ps.get_data_parallel_world_size()

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def broadcast_params(self):
        for param in self.module.parameters():
            dist.broadcast(tensor=param.data,
                           src=self.src_rank,
                           group=self.group)
            
    def allreduce_gradients(self):
        for param in self.module.parameters():
            dist.all_reduce(tensor=param.grad.data,
                            op=dist.ReduceOp.SUM,
                            group=self.group)
            param.grad.data /= self.world_size