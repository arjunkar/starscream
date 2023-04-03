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


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    test(2, 4)


if __name__ == "__main__":

    world_size = 16
    
    processes = []
    mp.set_start_method("fork")

    print_lock = mp.Lock()

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size))
        processes.append(p)
        
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("Passed!")