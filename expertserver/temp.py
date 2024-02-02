import torch
import os
import torch.distributed as dist
from workers import CPUWorker
from threading import Thread


def setup(args):
    dist.init_process_group(
        backend='nccl',
        init_method = 'env://', #f'{args.dist_url}:{args.port}', #'env://', #'tcp://localhost:12355',
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()

def distributed_fn(rank, main_fn, args, comm):
    
    ## Each Processes rank
    args.rank = args.node_rank * args.ngpu + rank
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    setup(args)
    
    print(f'rank: {args.rank}, world_size: {args.world_size}')
    
    main_fn(args, comm)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def multi_process_test(args, execute_fn):
    
    if args.nodes > 1:
        args.ngpu = torch.cuda.device_count()
        args.world_size = args.ngpu * args.nodes
    else:
        args.ngpu = args.world_size = torch.cuda.device_count()
    
    print(args.ngpu)
    
    os.environ['NCCL_SOCKET_IFNAME'] = "eno1np0"
    os.environ['MASTER_ADDR'] = args.dist_url
    os.environ['MASTER_PORT'] = str(args.port)
    
    if not args.moe_cpu:
        torch.multiprocessing.spawn(
                fn=distributed_fn,
                args=(execute_fn, args),
                nprocs=args.ngpu,
                join=True
            )
    else:
        cpu_worker = CPUWorker(args)
        #args.comm = 
        def _spawn(_distribute_fn):
            torch.multiprocessing.spawn(
                fn=_distribute_fn,
                args = (execute_fn, args, cpu_worker.comm),
                nprocs = args.ngpu
            )
        spawn_thread = Thread(target=_spawn, args=(distributed_fn,))
        spawn_thread.start()
        cpu_worker.start()

    return