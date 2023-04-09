"""
Constructing process groups for tensor model, pipeline model, and data parallelism.
Largely based on the Megatron parallel_state.py:
https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py

When we initialize Starscream, we create world_size processes and imagine each one
is sent to a different GPU.
To facilitate communication, torch.distributed is used to scatter, gather, reduce,
and perform other operations on groups of processes.
For instance, if we want to sum over the grads of all tensors in a certain subset
of GPUs, we need a torch.distributed process group for that subset.
Each process in our world_size will belong to a tensor model group, a pipeline
model group, and a data group.
The role of the functions in this file is to set up all of those groups for a given
process.
That is to say, all processes will run this code, and the output must depend on
process-specific properties accessed using primitives like torch.distributed.get_rank().

Let world_size: int = W.
The initialization consists of several integers:
tensor_model_parallel_size: int = T
    How many processes will contain shards of a single tensor in the model.
pipeline_model_parallel_size: int = P
    How many stages the model will be split over.
data_parallel_size: int = D = W // (T * P)
    How many microbatches can be computed simultaneously.
    For efficiency, we take D = W // (T*P) which effectively
    maximizes data parallelism subject to the selected T and P.
    However, data parallelism is a distinct concept.

Let us give some examples of these with increasing complexity.
0) No parallelism: P = D = T = 1, W = 1
    With only one process (GPU), the entire model must live on a single
    machine and process data from that machine only, without replication.
    For this case, torch.distributed is not necessary.
1) Pure data parallel: T = P = 1, W = D
    If we have no tensor or pipeline parallelism, the entire model is broadcast
    to every GPU and every GPU can compute a separate microbatch.
    We need only accumulate the gradients using an all_reduce.
    If we have more microbatches than GPUs in a single batch, we will need to
    do some extra processing in the optimizer and store various gradients
    or momenta in auxiliary buffers.
2) Pure tensor model parallel: P = D = 1, W = T
    With only tensor parallelism, the model is split tensor-wise across all
    GPUs ("sharded").
    Every GPU contains a portion (~1/W) of every layer in the model.
    The batch is then broadcast to every GPU and the full forward pass occurs
    in parallel.
3) Pure pipeline model parallel: D = T = 1, W = P
    With only pipeline parallelism, the model is split ("staged") into
    W stages and each GPU is responsible for a different stage.
    The microbatch moves forward stage by stage; when it reaches the end of
    one stage, it is sent point-to-point to the next one.
    During the backward pass, this communication is reversed.
    Note that we have used the term microbatch because there is an obvious
    form of "data parallelism" in the pipeline: when one stage finishes with
    a microbatch, it could be available to process the next one (depending on
    the precise pipeline schedule).
    However, this data parallelism is trivial because there are no tradeoffs
    associated with it -- if we use a pipeline at all, we might as well get
    the most out of it.
    For true data parallelism, we need D > 1 and multiple GPUs must contain
    the same model data so we can truly process multiple microbatches in
    parallel rather than just having multiple microbatches in the pipeline at
    the same time.
4) Data + tensor parallel: P = 1, W = D * T
    When P = 1, we have no pipeline.
    Instead, each model layer is sharded over T GPUs and we process D
    microbatches simultaneously.
5) Data + pipeline parallel: T = 1, W = P * D
    When T = 1, the model layers are not sharded individually.
    Instead, the model is staged into P stages with each stage fully on
    a single GPU, and this stage is replicated on D GPUs for data parallelism.
6) Tensor + pipeline parallel: D = 1, W = T * P
    When D = 1, we have no data parallelism, so there is only a single
    copy of any given weight in the model.
    We stage the model into P stages and each stage is sharded across T GPUs.
    A given batch is sent through stage by stage, each time passing through
    T GPUs containing the sharded layers of that stage.
7) Tensor + data + pipeline parallel: D = W // (T * P)
    When all forms of parallelism are active, W is partitioned three times.
    Let's consider the example W = 16 with T = 2 and P = 4 (so D = 2).
    We have GPUs [g0, ... , g15] each running a process.
    The number following g is the global "rank" of the process.
    The D = 2 data parallel size means we will split these 16 GPUs into two
    groups which will hold two identical copies of the model.
    We can think of this split as a pairing of the GPUs where a pair contain
    the same weights/layers.
    W // D data groups: 
    [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
    So e.g. g0 and g2 contain identical parameter information.
    Let us focus on one of these two groups, specifically [g0, g1, g4, g5, g8, g9, g12, g13].
    These 8 GPUs contain a single copy of the model.
    With P = 4, we need to stage the model into 4 stages.
    We group these stages together into W // P = 4 pipeline groups of P = 4, 
    so a single pipeline group represents P GPUs each associated 
    with a different stage of the pipeline.
    The pipeline groups are [g0, g4, g8, g12], [g1, g5, g9, g13].
    For the other data parallel set: [g2, g6, g10, g14], [g3, g7, g11, g15].
    In each stage, we will have T = 2 GPUs that contain layers which have been split
    in half, into two shards.
    We package these T GPUs into their own tensor model parallel group.
    The W // T = 8 tensor groups are [g0, g1], [g4, g5], [g8, g9], [g12, g13] and
    for the other data parallel set: [g2, g3], [g6, g7], [g10, g11], [g14, g15].
    The groups we have just described are the torch.distributed process groups
    we will create in this file.
"""

import torch.distributed as dist
from datetime import timedelta

# Tensor (sharded within a single layer) model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None

# Pipeline (staged into stages) model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None

# Model parallel group (both tensor and pipeline) that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None

# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None

# A list of global ranks for each tensor group to ease calculation of the source
# rank when broadcasting from the first tensor shard.
_TENSOR_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
_DATA_PARALLEL_GLOBAL_RANKS = None

def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1
) -> None:
    """
    initialize_model_parallel(T, P) creates the
    process groups described above.
    """
    assert dist.is_initialized(), "torch.distributed is not initialized."

    T = tensor_model_parallel_size
    P = pipeline_model_parallel_size
    W = dist.get_world_size()

    assert W % (T * P) == 0, "Product of model parallel sizes do not divide world size."

    # data_parallel_size
    D = W // (T * P)

    # Number of process groups of each type
    # num_D_groups = W // D
    # num_P_groups = W // P
    # num_T_groups = W // T

    # Global rank of the process running this code
    rank = dist.get_rank()

    # Generate tensor model parallel group
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_GLOBAL_RANKS
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        "Tensor model parallel group is already initialized."
    for rank_ in range(W):
        ranks = generate_tensor_model_parallel_group(rank_, T)
        group = dist.new_group(ranks, timeout=timedelta(seconds=60.))
        if rank in ranks:
            # Initialize tensor model parallel group
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_GLOBAL_RANKS = ranks

    # Generate pipeline model parallel group
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert _PIPELINE_MODEL_PARALLEL_GROUP is None, \
        "Pipeline model parallel group is already initialized."
    for rank_ in range(W):
        ranks = generate_pipeline_model_parallel_group(rank_, P, W)
        group = dist.new_group(ranks, timeout=timedelta(seconds=60.))
        if rank in ranks:
            # Initialize pipeline model parallel group
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks

    # Generate data parallel group
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GLOBAL_RANKS
    assert _DATA_PARALLEL_GROUP is None, \
        "Data parallel group is already initialized."
    for rank_ in range(W):
        ranks = generate_data_parallel_group(rank_, P, T, W)
        group = dist.new_group(ranks, timeout=timedelta(seconds=60.))
        if rank in ranks:
            # Initialize data parallel group
            _DATA_PARALLEL_GROUP = group
            _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Generate model parallel group
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, \
        "Model parallel group is already initialized."
    for rank_ in range(W):
        ranks = generate_model_parallel_group(rank_, P, T, W)
        group = dist.new_group(ranks, timeout=timedelta(seconds=60.))
        if rank in ranks:
            # Initialize model parallel group
            _MODEL_PARALLEL_GROUP = group


# The pattern appearing in the initialization, where we loop over all
# ranks in the global process group and re-compute the relevant groups
# for every rank and subsequently only update the global variables if we
# find the right group, may be confusing.
# Why did we not simply compute the groups directly given the rank and
# then make the groups once for each process?
# 
# The answer is related to the behavior of torch.distributed.new_group.
# This function requires all processes to run it, regardless of whether
# or not the process is contained in the group to be created.
# So if we tried to call new_group in two different processes with different
# groups to be created, we may deadlock as both new_group calls will be waiting
# for the other caller to check in.
# As such, it is recommended 
# (by the tech lead of PyTorch's Distributed team:
# https://github.com/pytorch/pytorch/issues/25390) 
# to have identical sequences of new_group calls, including identical arguments, 
# on every process.
# This leads to much wasted work but avoids deadlock, and is the strategy
# employed in NVIDIA's Megatron.


def generate_tensor_model_parallel_group(rank, T):
    """Compute tensor model parallel group of rank"""
    # If T = 2, global ranks range(W) correspond to tensor model parallel groups:
    # group(range(W)) = [0, 0, 1, 1, 2, 2, 3, 3, ...]
    start_rank = (rank // T) * T
    end_rank = start_rank + T
    return range(start_rank, end_rank)


def generate_pipeline_model_parallel_group(rank, P, W):
    """Compute pipeline model parallel group of rank"""
    # If W // P = 4, global ranks range(W) correspond to pipeline model parallel groups:
    # group(range(W)) = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, ...]
    num_P_groups = W // P
    offset = rank % num_P_groups
    return range(offset, W, num_P_groups)


def generate_data_parallel_group(rank, P, T, W):
    """Compute data parallel group of rank"""
    # If W // P = 4 and T = 2, global ranks range(W) correspond to data parallel groups:
    # group(range(W)) = [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, ...].
    # As another example, if W // P = 6 and T = 3, we have
    # group(range(W)) = [0, 1, 2, 0, 1, 2, 3, 4, 5, 3, 4, 5, ...].
    # Finally, if W // P = 6 and T = 2, we have
    # group(range(W)) = [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, ...].
    num_P_groups = W // P
    start_rank = (rank // num_P_groups) * num_P_groups
    end_rank = start_rank + num_P_groups
    offset = (rank - start_rank) % T
    return range(start_rank + offset, end_rank, T)


def generate_model_parallel_group(rank, P, T, W):
    """Compute model parallel group of rank"""
    # In the examples above, the model parallel groups are:
    # W // P = 4, T = 2:
    # group(range(W)) = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, ...]
    # W // P = 6, T = 3:
    # group(range(W)) = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, ...]
    # W // P = 6, T = 2:
    # group(range(W)) = [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, ...]
    num_P_groups = W // P
    block = (rank % num_P_groups) // T
    ranks = []
    # Unfortunately, this loop seems difficult to eliminate with arithmetic
    for p in range(P):
        ranks += range(p * num_P_groups + block * T, 
                       p * num_P_groups + (block+1) * T)
    return ranks


# The main logic of this file is contained in the generate_* functions above.
# The functions below allow us to access the various groups for a given rank 
# defined in that function while working in other files.
# They are boilerplate essentially copied from Megatron.


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    return not (_TENSOR_MODEL_PARALLEL_GROUP is None or
                _PIPELINE_MODEL_PARALLEL_GROUP is None or
                _DATA_PARALLEL_GROUP is None
            )


def get_tensor_model_parallel_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'Tensor model parallel group is not initialized.'
    return _TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        'Pipeline model parallel group is not initialized.'
    return _PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'Data parallel group is not initialized.'
    return _DATA_PARALLEL_GROUP


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, \
        'Model parallel group is not initialized.'
    return _MODEL_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'Tensor model parallel group is not initialized.'
    return dist.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    assert _PIPELINE_MODEL_PARALLEL_GROUP is not None, \
        'Pipeline model parallel group is not initialized.'
    return dist.get_world_size(group=get_pipeline_model_parallel_group())


def get_data_parallel_world_size():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, \
        'Data parallel group is not initialized.'
    return dist.get_world_size(group=get_data_parallel_group())


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return dist.get_rank(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    return dist.get_rank(group=get_pipeline_model_parallel_group())


def get_tensor_parallel_src_rank():
    """Return first rank in tensor model parallel group."""
    assert _TENSOR_GLOBAL_RANKS is not None, \
        'Tensor model parallel group is not initialized.'
    return _TENSOR_GLOBAL_RANKS[0]


def get_data_parallel_src_rank():
    """Return first rank in data parallel group."""
    assert _DATA_PARALLEL_GLOBAL_RANKS is not None, \
        'Data parallel group is not initialized.'
    return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return first rank in pipeline model parallel group."""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        'Pipeline model parallel group is not initialized.'
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return last rank in pipeline model parallel group."""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        'Pipeline model parallel group is not initialized.'
    return _PIPELINE_GLOBAL_RANKS[-1]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized."
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline."""
    assert _PIPELINE_GLOBAL_RANKS is not None, \
        "Pipeline parallel group is not initialized."
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]


def is_pipeline_first_stage():
    """Return True if in the first pipeline model parallel stage, False otherwise."""
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage():
    """Return True if in the last pipeline model parallel stage, False otherwise."""
    return get_pipeline_model_parallel_rank() == (
        get_pipeline_model_parallel_world_size() - 1)


def destroy_model_parallel():
    """Set the groups to none."""
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None

    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None

    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None

    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None