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
from schedules import GPipe, PipeDream


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


def test(t, p):

    rank = dist.get_rank()

    ps.initialize_model_parallel(tensor_model_parallel_size=t,
                                 pipeline_model_parallel_size=p)

    bsz = 3
    isz = 2
    num_classes = isz
    num_mb = 8

    model = nn.Linear(num_classes, num_classes)
    loader = FakeLoader(bsz, isz, num_classes)
    loss = nn.CrossEntropyLoss()

    sched = PipeDream(model, loader, loss, num_mb)
    sched.train_step()

    # Broadcast totaled input over microbatches from first stage
    if ps.is_pipeline_first_stage():
        inputs = [sched.input_store[i] for i in range(1, num_mb+1)]
    else:
        # Allocate memory in other stages
        inputs = [loader.__next__()[0] for _ in range(num_mb)]
    total_input = torch.cat(inputs, dim=0).detach()
    dist.broadcast(tensor=total_input,
                   src=ps.get_pipeline_model_parallel_first_rank(),
                   group=ps.get_pipeline_model_parallel_group()
                   )
    # Broadcast totaled answers over microbatches from last stage
    if ps.is_pipeline_last_stage():
        ans = [sched.input_store[-i] for i in range(1, num_mb+1)]
    else:
        # Allocate memory in other stages
        ans = [loader.__next__()[1] for _ in range(num_mb)]
    total_ans = torch.cat(ans, dim=0)
    dist.broadcast(tensor=total_ans,
                   src=ps.get_pipeline_model_parallel_last_rank(),
                   group=ps.get_pipeline_model_parallel_group()
                   )
    # Gather model
    weights = [torch.empty_like(model.weight.data) for _ in range(p)]
    biases = [torch.empty_like(model.bias.data) for _ in range(p)]
    dist.all_gather(tensor_list=weights,
                    tensor=model.weight.data,
                    group=ps.get_pipeline_model_parallel_group())
    dist.all_gather(tensor_list=biases,
                    tensor=model.bias.data,
                    group=ps.get_pipeline_model_parallel_group())
    for k in range(len(weights)):
        weights[k].requires_grad_(True)
        biases[k].requires_grad_(True)
    # Forward pass
    total_output = total_input.requires_grad_(True)
    for k in range(len(weights)):
        total_output = torch.matmul(total_output, weights[k].t()) + biases[k]
    total_loss = loss(total_output, total_ans) / (num_mb * bsz)
    # Backward pass
    total_loss.backward()
    # Check gradients
    local_rank = dist.get_rank(group=ps.get_pipeline_model_parallel_group())
    assert(
        torch.allclose(model.weight.grad, weights[local_rank].grad)
    )
    assert(
        torch.allclose(model.bias.grad, biases[local_rank].grad)
    )

    # Writing output, uncomment for testing
    # out_file = './starscream/log/'+str(rank)+'.out'
    # try:
    #     os.remove(out_file)
    # except OSError:
    #     pass
    # sys.stdout = open(out_file, 'w')
    # print("Input store:\n")
    # for key in sched.input_store.keys():
    #     print("Key ", key, "\n", sched.input_store[key])
    # print("\nOutput store:\n")
    # for key in sched.output_store.keys():
    #     print("Key ", key, "\n", sched.output_store[key])
    # print("\nWeight data:\n", model.weight.data)
    # print("\nGathered weight:\n", weights[local_rank])
    # print("\nWeight grad:\n", model.weight.grad)
    # print("\nGathered weight grad:\n", weights[local_rank].grad)
    # print("\nBias data:\n", model.bias.data)
    # print("\nGathered bias:\n", biases[local_rank])
    # print("\nBias grad:\n", model.bias.grad)
    # print("\nGathered bias grad:\n", biases[local_rank].grad)
    # print("\nTotal input:\n", total_input)
    # print("\nTotal answers:\n", total_ans)

    # Assertions complete
    ps.destroy_model_parallel()
    dist.barrier()


def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    t = 1
    p = 4
    test(t, p)


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