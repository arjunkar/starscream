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

def test(micro_batch_size,
         input_size,
         output_size,
         tensor_model_parallel_size,
         pipeline_model_parallel_size):

    ps.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size)
    
    group = ps.get_data_parallel_group()
    world_size = ps.get_data_parallel_world_size()
    
    linear = nn.Linear(input_size, output_size)
    linear_ddp = DDP(linear)
    x = torch.randn(size=(micro_batch_size, input_size), requires_grad=True)
    
    linear_ddp.broadcast_params()

    # test broadcast_params
    all_weight = [torch.empty_like(linear.weight.data) for _ in range(world_size)]
    all_bias = [torch.empty_like(linear.bias.data) for _ in range(world_size)]
    dist.all_gather(tensor_list=all_weight,
                    tensor=linear.weight.data,
                    group=group)
    dist.all_gather(tensor_list=all_bias,
                    tensor=linear.bias.data,
                    group=group)
    for i in range(world_size-1):
        assert(
            torch.allclose(all_weight[i], all_weight[i+1])
        )
        assert(
            torch.allclose(all_bias[i], all_bias[i+1])
        )

    output = linear_ddp(x)
    (output.sum() / micro_batch_size).backward()

    linear_ddp.allreduce_gradients()

    # test allreduce_gradients
    single_weight = linear_ddp.module.weight.data.clone().detach().requires_grad_(True)
    single_bias = linear_ddp.module.bias.data.clone().detach().requires_grad_(True)
    input_list = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(input_list, x, group=group)
    total_input = torch.cat(input_list, dim=0)
    output = torch.matmul(total_input, single_weight.t()) + single_bias
    (output.sum() / (world_size * micro_batch_size)).backward()

    assert(
        torch.allclose(linear_ddp.module.weight.grad,
                       single_weight.grad)
    )
    assert(
        torch.allclose(linear_ddp.module.bias.grad,
                       single_bias.grad)
    )

    ps.destroy_model_parallel()
    dist.barrier()


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    
    micro_batch_size = 3
    input_size = 5
    output_size = 12
    P = 2
    for T in range(1, 4):
        test(micro_batch_size,
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