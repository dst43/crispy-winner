import torch
import argparse
import nvtx
import torch.distributed as dist
import os
import torch.nn as nn
from storage_memory_utils import see_memory_usage

from fairseq.modules.transformer_layer import FeedForwardNetwork
from experts_cpuadam import Experts_CPUAdam
from torch.optim.adam import Adam
## Baseline
from omegaconf import OmegaConf
from fairseq import optim
from experts_cpu import Experts
from time import sleep
from fairseq.optim.dynamic_loss_scaler import DynamicLossScaler
import builtins

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
        self.count += n
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class TempModule(torch.nn.Module):
    def __init__(self, embed_dim):
        super(TempModule, self).__init__()
        self.pre_head = torch.nn.Linear(embed_dim, embed_dim)
        self.post_head = torch.nn.Linear(embed_dim, 1)

import expertserver.utils

def main(args):
    if args.rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass
    
    torch.cuda.synchronize()
    see_memory_usage(args.rank)
    input_dim = int(args.batch_size * args.seq_length * args.expert_capacity)
    # ## lazy init should be distributed
    experts = Experts(args, args.rank, FeedForwardNetwork, layer=0)
    
    args.expert_layer_cls = FeedForwardNetwork
    experts.update_freq = args.update_freq
    
    
    print('---------------- First Init -----------------')
    # for param in experts.expert_list[0].parameters():
    #     print(param.norm())
    #     break
    
    experts.lazy_init()
    
    print('---------------- Lazy Init -----------------')
    # for param in experts.expert_list[0].parameters():
    #     print(param.norm())
    #     break
    
    num_local_experts = len(experts.expert_list) // args.world_size

    experts_optimizer = Experts_CPUAdam(experts.expert_list, args.opt_type, args.rank, 0, args, FeedForwardNetwork, lr = 1)
    experts_optimizer.optim_event.wait()
    experts_optimizer.scaler = DynamicLossScaler()
    print('---------------- Optim Init -----------------')
    # for param in experts.expert_list[0].parameters():
    #     print(param.norm())
    #     break
    
    # experts_optimizer.pre_define_optim_states(args.rank, 0, args.moe_expert_count)
    
    print('---------------- Optim Pre Init -----------------')
    # for param in experts.expert_list[0].parameters():
    #     print(param.norm())
    #     break 
    
    experts.set_optimizer(experts_optimizer)
    
    gpu_expert_assign = [[0, 1], [2, 3]] #[[0, 1], [2, 3]]#[[0, 1, 2, 3]]
    iterations = args.iterations
    
    throughput_meter = AverageMeter('training_speed')
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    outputs = [0]#[torch.randn(input_dim, args.embed_dim, dtype=torch.float16).to('cuda') for _ in range(iterations)]
    
    # for expert in experts.expert_list:
    #     print('Initialization', 'param', args.rank, list(expert.parameters())[0].sum())
    
    linear = torch.nn.ModuleList([
        torch.nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim),
        torch.nn.Linear(args.decoder_embed_dim, 1)]
    ).to('cuda')
    linear.half()

    outputs[0]= torch.randn(input_dim, args.embed_dim, dtype=torch.float16).to('cuda')
    for i in range(args.epochs):
        # torch.cuda.empty_cache()
        
        # cuda_start.record()
        # torch.cuda.synchronize()
        
        for it in range(iterations):
            # see_memory_usage()
            # for expert in experts.expert_list:
            #     for p in expert.parameters():
            #         print('p.grad: ', end = "")
            #         if p.grad is not None:
            #             print(p.grad.device, p.grad.shape, p.grad.norm())
            #         else:
            #             print(p.grad)
            #         print('p._cpu_grad: ', end = "")
            #         if p._cpu_grad is not None:
            #             print(p._cpu_grad.device, p._cpu_grad.shape, p._cpu_grad.norm())
            #         else:
            #             print(p._cpu_grad, end = "")
        
            ## Explicitly set grad to None since we don't have any params in experts but do have expert list
            main_stream = torch.cuda.current_stream()
            
            first_comm_stream = experts._streams['communication']
            first_comm_stream.wait_stream(main_stream)
            #with torch.cuda.stream(first_comm_stream):
            
            with nvtx.annotate('wait prev optim'):
                print('wait for prev optim')
                experts._wait_for_previous_optim_step()
            first_exp_id = gpu_expert_assign[args.rank][0]
            #print(f'first_exp_id: {first_exp_id}')
            experts._upload_params(experts.expert_list[first_exp_id].parameters())
            
            first_comp_stream = experts.comp_streams[first_exp_id]
            first_comp_stream.wait_stream(main_stream)
            with torch.cuda.stream(first_comp_stream):
                output = outputs[0].chunk(num_local_experts, dim=0)
            
            print('forward')
            # output = non_experts.pre_head(output)
            see_memory_usage()
            # print(torch.cuda.memory_allocated() / 1024 / 1024)
            torch.cuda.mem_get_info()
            with nvtx.annotate(f"forward{i}", color="blue"):
                output, last_comp_stream = experts(output, gpu_expert_assign, rank = args.rank)
            #output = non_experts.post_head(output)
                # print(torch.cuda.memory_allocated() / 1024 / 1024)
                
                with torch.cuda.stream(last_comp_stream):
                    output = torch.cat(output, dim = 0)
                    def _backward_stream_connection(*unsued):
                        last_comp_stream.wait_stream(main_stream)
                    output.register_hook(_backward_stream_connection)
                
                ## Forward Main Stream should wait for the last comp_stream
                main_stream.wait_stream(last_comp_stream)
                for layer in linear:
                    output = layer(output)
                
                # print(torch.cuda.memory_allocated() / 1024 / 1024)
                
                def _backward_pre_expert_upload(*unused):
                    last_exp_id = gpu_expert_assign[args.rank][-1]
                    # print(f'last_exp_id: {last_exp_id}', gpu_expert_assign[args.rank])
                    experts._streams['communication'].wait_stream(main_stream)
                    experts._upload_params(experts.expert_list[last_exp_id].parameters())
                
                output.register_hook(_backward_pre_expert_upload)
                
                # main_stream.wait_stream(comp_connection_stream)
                
                loss = torch.log(output.norm(dim = 0))
                
                # print(torch.cuda.memory_allocated() / 1024 / 1024)

                # torch.cuda.synchronize()
                
            # loss_fn = torch.nn.CrossEntropyLoss()
            # loss = loss_fn(output, torch.ones(size=(args.batch_size * args.seq_length,), device='cuda', dtype=torch.float16))
            # print('loss', loss)
            # torch.cuda.synchronize()

            ## Backward
            with nvtx.annotate(f"backward{i}", color="red"):
                print('backward')
                loss.backward()
                # torch.cuda.sync
                # hronize()

        #print(torch.cuda.memory_snapshot())
        
            ## Need to know whether current backward call has overflow so the step has been skipped or not
            # if not experts_optimizer.overflow:
            #     experts_optimizer.increment_step(gpu_expert_assign[args.rank])
            #experts.optimize()
            # if it == iterations - 1:
            #     torch.cuda.synchronize()
            #     for exp_id in range(num_local_experts):
            #         experts.optimizer.step(exp_id)
            # print('optimize')
            ## Non-experts Optimize // Experts Optimize call happend in the post_gradient_hook
            # non_experts_optimizer.step()
            
            ## Non-experts Zero Grad // Experts Zero Grad called with step() Experts_CPUAdam.step(exp_id)
            # non_experts_optimizer.zero_grad()
        ## Finalize the Optimization Step
        # for t in experts_optimizer.thread_list:
        #     if t:
        #         t.join()
        #experts._wait_for_previous_optim_step()
        for exp_id, thread in enumerate(experts_optimizer.thread_list):
            if thread:
                print(f'{exp_id} thread joining')
                thread.join()
                experts_optimizer.optim_expert_events[exp_id].wait()
                experts_optimizer.set_grad_to_none(exp_id)
        print('the last join finished')
        # cuda_end.record()
        # torch.cuda.synchronize()
        # for expert in experts.expert_list:
        #     print('param', args.rank, list(expert.parameters())[0].sum())
        
        # for exp_id in  range(len(experts_optimizer.optimizers)):
        #     optim = experts_optimizer.optimizers[0]
        #     for group_id, group in enumerate(optim.param_groups):
        #         for param_id, p in enumerate(group['params']):
        #             state = optim.state[p]
        #             print('exp_avg', args.rank, state['exp_avg'].sum())
        #             print('exp_avg_sq', args.rank, state['exp_avg_sq'].sum())
        #             break
        # if i > 0:
        #     throughput_meter.update(cuda_start.elapsed_time(cuda_end))
        
    experts_optimizer.terminate()
    print(throughput_meter)
    
def main_baseline(args):
    input_dim = int(args.batch_size * args.seq_length * args.expert_capacity)
    experts = nn.ModuleList([FeedForwardNetwork(args, args.embed_dim, args.ffn_dim) for _ in range(args.moe_expert_count)]).to('cuda')
    experts.half()
    #non_experts = TempModule(args.embed_dim).to('cuda')
    #non_experts.half()
    cfg_dls = OmegaConf.create(
            {
                "optimization": {
                    "lr": [0.1],
                },
                "optimizer": {
                    "_name": "adam",
                    "lr": [0.1],
                    "adam_betas": "(0.9, 0.999)",
                    "adam_eps": 1e-8,
                    "weight_decay": 0.0,
                },
                "common": {
                    "fp16_init_scale": 1,
                    "fp16_scale_window": 1,
                    "fp16_scale_tolerance": 1,
                    "threshold_loss_scale": 1,
                    "min_loss_scale": 1e-4,
                    "tpu": False,
                },
            }
        )
    
    optimizer = optim.FP16Optimizer.build_optimizer(cfg_dls, list(experts.parameters()))
    # storage_memory_utils.see_memory_usage()
    iterations = args.iterations
    num_local_experts = args.moe_expert_count
    
    throughput_meter = AverageMeter('training_speed')
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    outputs = [0]#[torch.randn(input_dim, args.embed_dim, dtype=torch.float16).to('cuda') for _ in range(iterations)]
    for i in range(args.epochs):
        torch.cuda.empty_cache()
        outputs[0] = torch.randn(input_dim, args.embed_dim, dtype=torch.float16).to('cuda')
        cuda_start.record()
        torch.cuda.synchronize()
        for it in range(iterations):
            
            ## Forward
            # print('forward')
            #output = non_experts.pre_head(output)
            
            output_list = []
            output = outputs[0].chunk(num_local_experts, dim=0)
            for chunk, expert in zip(output, experts):
                output_list.append(expert(chunk))
            #output = non_experts.post_head(output)
            output = torch.cat(output_list, dim = 0)

            loss = output.sum(dim = 0).sum(dim = 0)
            
            #see_memory_usage()
            
            ## Backward
            # print('backward')
            # torch.cuda.synchronize()
            # see_memory_usage()
            
            optimizer.backward(loss)
            
            #see_memory_usage()
            
            # print('optimize')
            #if it == iterations - 1:
                #print(it)
            optimizer.step()
            optimizer.zero_grad()
            
            #see_memory_usage()
        cuda_end.record()
        torch.cuda.synchronize()
        if i > 0:
            throughput_meter.update(cuda_start.elapsed_time(cuda_end))
    
    print(throughput_meter)
    
def setup(args):
    dist.init_process_group(
        backend='nccl',
        init_method = 'env://', #f'{args.dist_url}:{args.port}', #'env://', #'tcp://localhost:12355',
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    
def distributed_fn(rank, main_fn, args):
    
    ## Each Processes rank
    args.rank = args.node_rank * args.ngpu + rank
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    setup(args)
    
    print(f'rank: {args.rank}, world_size: {args.world_size}')
    # temp = queue.get()
    # print(id(temp), id(args), rank)
    # del temp
    
    main_fn(args)
    
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
    
    # #kwargs = torch.randn(100, 1000).share_memory_()
    # print(id(kwargs), id(args), end='\n\n')
    
    # manager = torch.multiprocessing.Manager()
    # queue = manager.Queue()
    # queue.put(kwargs)
    
    #torch.multiprocessing.set_start_method('fork')
    torch.multiprocessing.spawn(
        fn=distributed_fn,
        args=(execute_fn, args),
        nprocs=args.ngpu,
        join=True,
    )
    return

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    ## Training Args ##
    parser.add_argument('--embed-dim', default=2048, type=int)
    parser.add_argument('--seq-length', default=2048, type=int)
    parser.add_argument('--batch-size', default = 2, type=int)
    
    parser.add_argument('--expert-capacity', default=1.0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--moe-expert-count', default=4, type=int)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--opt-type', default='torch', type=str)
    parser.add_argument('--update-freq', default=1, type=int)
    
    ## Distributed Args ##
    parser.add_argument('--nodes', default=1, type=int)
    parser.add_argument('--node-rank', default=0, type=int, help='[0, nodes-1]')
    parser.add_argument('--dist-url', default='127.0.0.1', type=str)
    parser.add_argument('--port', default=12345, type=str)
    
    ## 
    args = parser.parse_args()
    args.decoder_embed_dim = args.embed_dim
    args.decoder_ffn_embed_dim = args.ffn_dim = args.embed_dim * 4
    args.fp16 = True
    args.moe_cpu = True
    args.tokens_per_sample = args.seq_length
    multi_process_test(args, main)
    
    #main_baseline(args)