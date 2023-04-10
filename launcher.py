"""
Starscream launcher combining data, pipeline, and tensor parallelism
in the training step of a fully connected network.
"""


import os
import sys
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import parallel_state as ps
from distributed import DistributedDataParallel as DDP
from model import ParallelMLP
from schedules import GPipe, PipeDream
from loader import ParallelLoader
from optim import ParallelSGDOptimizer


def train(model, loader, num_micro_batches):

    loss = nn.CrossEntropyLoss()

    sched = PipeDream(model=model, 
                  data_loader=loader, 
                  loss_fn=loss, 
                  num_micro_batches=num_micro_batches)
    
    optim = ParallelSGDOptimizer(model, sched, lr=5e-3)

    for _ in range(2):
        # Basic Starscream update loop
        optim.zero_grad()
        sched.train_step()
        model.allreduce_gradients()
        optim.step()

    # Writing output, uncomment for testing
    rank = dist.get_rank()
    out_file = './starscream/log/'+str(rank)+'.out'
    try:
        os.remove(out_file)
    except OSError:
        pass
    sys.stdout = open(out_file, 'w')
    # View shards of model
    print("Parameters:\n")
    for param in model.parameters():
        print(param)
    # View pipeline state
    print("Input store:\n")
    for key in sched.input_store.keys():
        print("Key ", key, "\n", sched.input_store[key])
    print("\nOutput store:\n")
    for key in sched.output_store.keys():
        print("Key ", key, "\n", sched.output_store[key])
    

def init_process(rank, 
                world_size,
                tensor_model_parallel_size,
                pipeline_model_parallel_size,
                num_layers,
                input_size,
                hidden_size,
                output_size,
                micro_batch_size,
                num_micro_batches):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    ps.initialize_model_parallel(tensor_model_parallel_size=tensor_model_parallel_size, 
                                 pipeline_model_parallel_size=pipeline_model_parallel_size)

    model = DDP(ParallelMLP(num_layers=num_layers, 
                            input_size=input_size, 
                            hidden_size=hidden_size, 
                            output_size=output_size))
    
    model.broadcast_params()
    
    loader = ParallelLoader(micro_bsz=micro_batch_size, 
                            input_sz=input_size, 
                            output_sz=output_size)
    
    train(model, loader, num_micro_batches)


if __name__ == "__main__":

    world_size = 8
    tensor_model_parallel_size = 2
    pipeline_model_parallel_size = 2
    num_layers = 2
    input_size = 3
    hidden_size = 2
    output_size = 4
    micro_batch_size = 3
    num_micro_batches = 4

    processes = []
    mp.set_start_method("spawn")

    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, 
                                                  world_size,
                                                  tensor_model_parallel_size,
                                                  pipeline_model_parallel_size,
                                                  num_layers,
                                                  input_size,
                                                  hidden_size,
                                                  output_size,
                                                  micro_batch_size,
                                                  num_micro_batches))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("Completed!")