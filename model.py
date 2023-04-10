"""
Basic parallel MLP model.

Employs tensor model parallelism with the ParallelLinear layer.
Will be staged under pipeline model parallelism with a Schedule.
"""

from torch import Tensor
import torch.nn as nn
import parallel_state as ps
from layers import ParallelLinear


class ParallelMLP(nn.Module):
    def __init__(self, 
                 num_layers: int, 
                 input_size: int, 
                 hidden_size: int,
                 output_size: int) -> None:
        super().__init__()
        assert ps.model_parallel_is_initialized(), \
            "Parallel state is not initialized."
        assert num_layers % ps.get_pipeline_model_parallel_world_size() == 0, \
            "num_layers must be divisible by pipeline_model_parallel_size."
        
        num_layers_on_stage = num_layers // ps.get_pipeline_model_parallel_world_size()
        d_in = hidden_size if not ps.is_pipeline_first_stage() else input_size
        d_out = hidden_size if not ps.is_pipeline_last_stage() else output_size
        layer_sizes = [d_in] + [hidden_size for _ in range(num_layers_on_stage-1)] + [d_out]
        self.layers = nn.ModuleList(
            [ParallelLinear(layer_sizes[i], layer_sizes[i+1]) for i in range(num_layers_on_stage)]
        )

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x