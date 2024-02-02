import torch
import torch.distributed as dist
import argparse
import os
import copy
import psutil
import time
from time import sleep

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def get_global_world_size():
    return torch.distributed.get_world_size()

def get_global_rank():
    return torch.distributed.get_rank()

def _find_my_group_index(grouped_ranks):
    my_rank = get_global_rank()
    for i, group in enumerate(grouped_ranks):
        if my_rank in group:
            return i
        
def get_global_group():
    if not hasattr(get_global_group, "_global_group"):
        # ideally we could use torch.distributed.group.WORLD, but it seems
        # to cause random NCCL hangs in some cases
        get_global_group._global_group = dist.new_group()
    return get_global_group._global_group

def get_all2all_group(moe_expert_count):
    if not hasattr(get_all2all_group, "_all2all_groups"):
        world_size = get_global_world_size()

        # more experts than world size
        if world_size <= moe_expert_count:
            assert moe_expert_count % world_size == 0
            all2all_groups = [[i for i in range(world_size)]]

        # larger world than num experts
        else:
            assert world_size % moe_expert_count == 0
            ranks_per_group = world_size // moe_expert_count
            all2all_groups = [[i * moe_expert_count + j for j in range(moe_expert_count)]
                                for i in range(ranks_per_group)]

        get_all2all_group._all2all_group_idx = all2all_groups
        get_all2all_group._all2all_groups = [dist.new_group(g) for g in all2all_groups]

    my_group_idx = _find_my_group_index(get_all2all_group._all2all_group_idx)
    return get_all2all_group._all2all_groups[my_group_idx]
    
def all_to_all_wrapper(args, input, output, a_to_a_cpu_meter, a_to_a_gpu_meter, \
                                            input_list = None, output_list = None):
    # always record times, since it is not a lot of overhead
    # if we do not log it we simply clear it off in record_all_to_all_stats
    ##################################
    torch.cuda.synchronize()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.time() * 1000
    cuda_start.record()
    ###################################
    ##### What you want to measure ####
    #print(f'rank:{args.rank} + Before: {input}, {output}')
    dist.all_to_all_single(output, input, \
                            output_split_sizes=output_list, \
                            input_split_sizes=input_list, \
                            group=get_all2all_group(args.world_size))
    #print(f'rank:{args.rank} + After: {input}, {output}')
    ###################################
    cuda_end.record()
    cpu_end = time.time() * 1000
    torch.cuda.synchronize()
    if a_to_a_cpu_meter:
        a_to_a_cpu_meter.update((cpu_end - cpu_start))
    
    if a_to_a_gpu_meter:
        a_to_a_gpu_meter.update(cuda_start.elapsed_time(cuda_end))
    ###################################
    
def main_fn(args):
    
    print(f'rank: {args.rank}, id: {os.getpid()}, parent_id: {psutil.Process(os.getpid()).ppid()}')
    
    input_dim = int(args.batch_size * args.seq_length * args.expert_capacity)
    equal_a_to_a_cpu_meter = AverageMeter('equal_cpu_all_to_all')
    equal_a_to_a_gpu_meter = AverageMeter('equal_gpu_all_to_all')
    unequal_a_to_a_cpu_meter = AverageMeter('unequal_cpu_all_to_all')
    unequal_a_to_a_gpu_meter = AverageMeter('unequal_gpu_all_to_all')
    
    ### NOTE ###
    ### output_split_list = input_split_list.T (transpose)
    input_split_container = []
    input_split_list = [input_dim // 2, input_dim // 4,  input_dim // 8,  input_dim // 8]
    for _ in range(args.world_size):
        input_split_container.append(copy.deepcopy(input_split_list))
        input_split_list.insert(0, input_split_list.pop())
    
    output_split_container = copy.deepcopy(input_split_container)
    output_split_container = list(map(list, zip(*output_split_container)))
    
    dummy_dtype = torch.float16 if args.fp16 else torch.float32
    
    print(input_dim, args.embed_dim)
    for i in range(args.epochs):
        dummy_all_to_all = torch.randn(input_dim, args.embed_dim, dtype=dummy_dtype).to('cuda')
        output = torch.zeros_like(dummy_all_to_all)
        torch.cuda.synchronize()
        # if i % 2 == 0:
        if i < 5:
            all_to_all_wrapper(args, dummy_all_to_all, output, None, None)
        else:
            all_to_all_wrapper(args, dummy_all_to_all, output, equal_a_to_a_cpu_meter, equal_a_to_a_gpu_meter)
    
        # else:            
        #     if i < 5:
        #         all_to_all_wrapper(args, dummy_all_to_all, output, None, None, \
        #                                 input_split_container[args.rank], output_split_container[args.rank])
        #     else:
        #         all_to_all_wrapper(args, dummy_all_to_all, output, unequal_a_to_a_cpu_meter, unequal_a_to_a_gpu_meter, \
        #                                 input_split_container[args.rank], output_split_container[args.rank])
        #assert dummy_all_to_all.sum() != output.sum()
        if i == args.epochs - 1:
            print(f'{i}th rank[{args.rank}] Equal, cpu meter: {equal_a_to_a_cpu_meter}, gpu meter: {equal_a_to_a_gpu_meter}, count: {equal_a_to_a_gpu_meter.count}')
            #print(f'{i}th rank[{args.rank}] UnEqual, cpu meter: {unequal_a_to_a_cpu_meter}, gpu meter: {unequal_a_to_a_gpu_meter}, count: {unequal_a_to_a_gpu_meter.count}')
    return

def all_reduce_test(args):
    randint = torch.randint(high = 10, size = (4, )).to('cuda')
    print(f"randint: {randint}")
    args.find_max = True
    if args.find_max:
        op = torch.distributed.ReduceOp.MAX
    else:
        op = torch.distributed.ReduceOp.SUM
    dist.all_reduce(randint, op = op)
    print(f'after reduce: {randint}')

def setup(args):
    dist.init_process_group(
        backend='nccl',
        init_method = 'env://', #f'{args.dist_url}:{args.port}', #'env://', #'tcp://localhost:12355',
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    
def distributed_comm(rank, main_fn, args):
    
    ## Each Processes rank
    args.rank = args.node_rank * args.ngpu + rank
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    setup(args)
    
    print(f'rank: {args.rank}, world_size: {args.world_size}')
    dist.all_reduce(torch.zeros(1).to('cuda'))
    
    main_fn(args)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier(get_global_group())

def comm_test(args, execute_fn):
    
    if args.nodes > 1:
        args.ngpu = torch.cuda.device_count()
        args.world_size = args.ngpu * args.nodes
    else:
        args.ngpu = args.world_size = torch.cuda.device_count()
    
    os.environ['NCCL_SOCKET_IFNAME'] = "eno1np0"
    os.environ['MASTER_ADDR'] = args.dist_url
    os.environ['MASTER_PORT'] = str(args.port)

    torch.multiprocessing.spawn(
        fn=distributed_comm,
        args=(execute_fn, args),
        nprocs=args.ngpu,
        join=True,
    )
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Model Network Args ##
    parser.add_argument('--embed-dim', default=2048, type=int)
    parser.add_argument('--ffn-dim', default=8192, type=int)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--seq-length', default=2048, type=int)
    parser.add_argument('--expert-capacity', default=1.0, type=float)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--epochs', default=50, type=int)
    
    ## Distributed Args ##
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--node-rank', default=0, type=int, help='[0, nodes-1]')
    parser.add_argument('--dist-url', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=12345, type=str)
    
    args = parser.parse_args()
    comm_test(args, all_reduce_test)