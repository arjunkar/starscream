"""
Multiprocessing test of parallel_state methods.
"""

import os
import torch.distributed as dist
import torch.multiprocessing as mp
import parallel_state as ps


def test(tensor_model_parallel_size,
         pipeline_model_parallel_size):
    
    rank = dist.get_rank()

    ps.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size)
    
    assert(rank in ps._DATA_PARALLEL_GLOBAL_RANKS)
    assert(rank in ps._PIPELINE_GLOBAL_RANKS)
    assert(rank in ps._TENSOR_GLOBAL_RANKS)

    ps.destroy_model_parallel()
    dist.barrier()


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    T = 1
    while T <= size:
        P = 1
        while T * P <= size:
            test(T, P)
            P *= 2
        T *= 2


if __name__ == "__main__":

    world_size = 8
    
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