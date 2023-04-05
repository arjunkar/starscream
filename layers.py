"""
A tensor model parallel linear layer for Starscream.

Inspired by Megatron's ColumnParallelLinear class:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py
"""

import torch
from torch import Tensor
from torch.autograd.function import FunctionCtx
from torch import nn
import torch.distributed as dist
import parallel_state as ps
from typing import Tuple


class ParallelLinearFunctional(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx: FunctionCtx,
                input: Tensor,
                weight: Tensor,
                bias: Tensor) -> Tensor:
        ctx.save_for_backward(input, weight)

        output = torch.matmul(input, weight) + bias

        group = ps.get_tensor_model_parallel_group()
        world_size = ps.get_tensor_model_parallel_world_size()
        output_list = [torch.empty_like(output) for _ in range(world_size)]
        # Gather output from all shards
        dist.all_gather(output_list, output, group=group)
        return torch.cat(output_list, dim=-1)
    
    @staticmethod
    def backward(ctx: FunctionCtx,
                 grad_output: Tensor) -> Tuple[Tensor]:
        # Expects [batch_dim, num_features] input.  Broadcasting not supported.
        input, weight = ctx.saved_tensors

        rank = ps.get_tensor_model_parallel_rank()
        group = ps.get_tensor_model_parallel_group()
        
        shard_size = weight.size()[-1]
        # Slicing grad_output to obtain relevant shard
        grad_output_shard = torch.split(grad_output, shard_size, dim=-1)[rank]
        
        grad_input = torch.matmul(grad_output_shard, weight.t())
        # Because the input tensor is shared among the tensor model
        # parallel group, we need to all_reduce its gradient to capture
        # contributions from all linear layer shards
        dist.all_reduce(grad_input, dist.ReduceOp.SUM, group=group)
        grad_weight = torch.matmul(input.t(), grad_output_shard)
        grad_bias = grad_output_shard.sum(dim=0)
        return grad_input, grad_weight, grad_bias


class ParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        assert ps.model_parallel_is_initialized(), \
            "Parallel state is not initialized."
        assert d_out % ps.get_tensor_model_parallel_world_size() == 0, \
            "Number of linear layer columns must be divisible by tensor model parallel world size."

        world_size = ps.get_tensor_model_parallel_world_size()
        shard_size = d_out // world_size

        # Each process in the tensor parallel group contains a shard of the full weight/bias
        self.weight = nn.Parameter(torch.empty(size=(d_in, shard_size)))
        self.bias = nn.Parameter(torch.empty(size=(shard_size,)))
        nn.init.uniform_(self.weight, -1/d_out, 1/d_out)
        nn.init.uniform_(self.bias, -1/d_out, 1/d_out)

    def forward(self, x: Tensor):
        output = ParallelLinearFunctional.apply(x, self.weight, self.bias)
        return output