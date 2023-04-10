"""
Distributed data loader for Starscream.
"""

import torch
import torch.distributed as dist
import parallel_state as ps


class ParallelLoader():
    def __init__(self,
                 micro_bsz: int,
                 input_sz: int,
                 output_sz: int) -> None:
        assert ps.model_parallel_is_initialized(), \
            "Parallel state is not initialized."
        assert ps.get_data_parallel_world_size() > 1, \
            "Data parallel size must be at least 2 to use ParallelLoader."

        self.micro_bsz = micro_bsz
        self.input_sz = input_sz
        self.output_sz = output_sz

    def __next__(self):
        if ps.get_tensor_parallel_src_rank() == dist.get_rank():
            # Load fake data into tensor parallel source
            input = torch.randn(size=(self.micro_bsz, self.input_sz))
            ans = torch.randint(low=0, 
                                high=self.output_sz, 
                                size=(self.micro_bsz,), 
                                dtype=torch.int64)
        else:
            # Allocate space on other tensor parallel nodes for broadcast.
            # Matches fake data for simplicity.
            input = torch.randn(size=(self.micro_bsz, self.input_sz))
            ans = torch.randint(low=0, 
                                high=self.output_sz, 
                                size=(self.micro_bsz,), 
                                dtype=torch.int64)
        dist.broadcast(tensor=input,
                       src=ps.get_tensor_parallel_src_rank(),
                       group=ps.get_tensor_model_parallel_group())
        dist.broadcast(tensor=ans,
                       src=ps.get_tensor_parallel_src_rank(),
                       group=ps.get_tensor_model_parallel_group())
        return (input, ans)
        
