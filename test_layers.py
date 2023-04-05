"""
Multiprocessing test of parallel neural network layers.
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import parallel_state as ps
import layers as ss


def test(batch_size,
         input_size,
         output_size,
         tensor_model_parallel_size,
         pipeline_model_parallel_size):

    rank = dist.get_rank()

    ps.initialize_model_parallel(
        tensor_model_parallel_size,
        pipeline_model_parallel_size)
    
    local_rank = ps.get_tensor_model_parallel_rank()
    group = ps.get_tensor_model_parallel_group()
    world_size = ps.get_tensor_model_parallel_world_size()
    
    linear = ss.ParallelLinear(input_size, output_size)
    x = torch.randn(size=(batch_size, input_size), requires_grad=True)
    dist.broadcast(tensor=x, 
                   src=ps.get_tensor_parallel_src_rank(), 
                   group=group
                   )
    output = linear(x)
    output.sum().backward()

    shard_size = output_size // world_size

    total_weight = [torch.empty(size=(input_size, shard_size)) 
                    for _ in range(world_size)
    ]
    total_bias = [torch.empty(size=(shard_size,))
                    for _ in range(world_size)
    ]
    dist.all_gather(tensor_list=total_weight, 
                    tensor=linear.weight.data, 
                    group=group)
    dist.all_gather(tensor_list=total_bias, 
                    tensor=linear.bias.data, 
                    group=group)
    
    total_weight = torch.cat(total_weight, dim=-1).detach().requires_grad_(True)
    total_bias = torch.cat(total_bias, dim=-1).detach().requires_grad_(True)
    total_input = x.clone().detach().requires_grad_(True)
    
    total_output = torch.matmul(total_input, total_weight) + total_bias
    total_output.sum().backward()

    assert(
        torch.allclose(total_output, output)
    )
    assert(
        torch.allclose(torch.split(total_weight.grad, 
                                   shard_size,
                                   dim=-1)[local_rank],
                        linear.weight.grad
        )
    )
    assert(
        torch.allclose(torch.split(total_bias.grad, 
                                   shard_size,
                                   dim=-1)[local_rank],
                        linear.bias.grad
        )
    )
    assert(
        torch.allclose(total_input.grad, x.grad)
    )

    # Writing output, uncomment for testing
    # out_file = './starscream/log/'+str(rank)+'.out'
    # try:
    #     os.remove(out_file)
    # except OSError:
    #     pass
    # sys.stdout = open(out_file, 'w')
    # print('Results of rank ', rank)
    # print('Input:\n', x)
    # print('Input copy:\n', total_input)
    # print('Input grads:\n', x.grad)
    # print('Total input grads:\n', total_input.grad)

    # print('Weight:\n', linear.weight)
    # print('Bias:\n', linear.bias)
    # print('Output:\n', output)
    # print('Weight grads:\n', linear.weight.grad)
    # print('Bias grads:\n', linear.bias.grad)

    # print('Total weight:\n', total_weight)
    # print('Total bias:\n', total_bias)
    # print('Total output:\n', total_output)
    # print('Total weight grads:\n', total_weight.grad)
    # print('Total bias grads:\n', total_bias.grad)

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