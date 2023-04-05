"""
Multiprocessing test of distributed data parallel wrapper.
"""

import os
import sys
import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp
from distributed import DistributedDataParallel as DDP
import parallel_state as ps

def test(batch_size,
         input_size,
         output_size,
         tensor_model_parallel_size,
         pipeline_model_parallel_size):

    rank = dist.get_rank()

    ps.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size)
    
    group = ps.get_data_parallel_group()
    world_size = ps.get_data_parallel_world_size()
    
    linear = nn.Linear(input_size, output_size)
    linear_ddp = DDP(linear)
    x = torch.randn(size=(batch_size, input_size), requires_grad=True)
    
    linear_ddp.broadcast_params()

    # test broadcast_params here

    output = linear_ddp(x)
    output.sum().backward()

    linear_ddp.allreduce_gradients()

    # test allreduce_gradients here

    ps.destroy_model_parallel()
    dist.barrier()


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    
    batch_size = 3
    input_size = 5
    output_size = 12
    P = 2
    for T in range(1, 4):
        test(batch_size,
            input_size,
            output_size,
            T, P)


if __name__ == "__main__":

    world_size = 12
    
    processes = []
    mp.set_start_method("spawn")
    # Changing the start method to "fork" will increase speed but
    # can lead to subtle errors.
    # For instance, a single test may succeed but a string of tests
    # together may lead to a connection reset.
    # I do not understand this behavior very well; some preliminary reading
    # suggests it may be related to the different ways "fork" and "spawn" handle
    # open ports and networking.

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size))
        processes.append(p)
        
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("Passed!")