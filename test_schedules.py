"""
Multiprocessing test of Starscream pipeline schedules.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import parallel_state as ps
from schedules import GPipe


class FakeLoader():
    def __init__(self, 
                 bsz: int,
                 isz: int,
                 num_classes: int) -> None:
        self.bsz = bsz
        self.isz = isz
        self.num_classes = num_classes

    def __next__(self):
        return ( 
            torch.randn(
                size=(self.bsz, self.isz)
            ),
            torch.randint(
                low=0,
                high=self.num_classes,
                size=(self.bsz,),
                dtype=torch.int64,
            )
        )


def test():

    rank = dist.get_rank()

    ps.initialize_model_parallel(1, 4)

    bsz = 3
    isz = 4
    num_classes = 4
    num_mb = 4

    model = nn.Linear(num_classes, num_classes)
    loader = FakeLoader(bsz, isz, num_classes)
    loss = nn.CrossEntropyLoss()

    sched = GPipe(model, loader, loss, num_mb)
    sched.train_step()

    # Writing output, uncomment for testing
    out_file = './starscream/log/'+str(rank)+'.out'
    try:
        os.remove(out_file)
    except OSError:
        pass
    sys.stdout = open(out_file, 'w')
    print("Input store:\n")
    for key in sched.input_store.keys():
        print("Key ", key, "\n", sched.input_store[key])
    print("\nOutput store:\n")
    for key in sched.output_store.keys():
        print("Key ", key, "\n", sched.output_store[key])
    print("\nWeight data:\n", model.weight.data)
    print("\nWeight grad:\n", model.weight.grad)
    print("\nBias data:\n", model.bias.data)
    print("\nBias grad:\n", model.bias.grad)


    dist.barrier()


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    test()


if __name__ == "__main__":

    world_size = 4
    
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