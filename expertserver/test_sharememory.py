## NOTE
## Multiprocessing Queue and Array are not
## doing well on torch.multiprocessing.spawn
## I might need to do work-around to share CPU model memory

## First, make the model on CPU and save it as .pt
## and following https://github.com/project-codeflare/zero-copy-model-loading/blob/main/notebooks/h5_poc.ipynb
## Since optimizer.step will be done inplace operation,
## there will be no memory address change.

## + Expert Parallelization
## + Expert Server on muti-Node
## + Linear-level parallelization(increasing granularity with expert division)
## - sharing I/O with multiple GPUs
## - sharing memory
##

## NOTE
## Even though we pinned a tensor,
## when it goes into the Queue, it became non pinned memory


import torch

#from multiprocessing import Array, Process
# from threading import Thread
# from experts_cpu import Experts
# from storage_memory_utils import see_memory_usage
# import cpu_adam
# import sys
# from utils import Communicator
# from ttemp import func_torch
# import argparse
import os
import numpy as np
import zipfile
import h5py


def func(arr, temp):
    print(id(arr), id(temp))
    
def optimizer_states_address_test():
    
    model = torch.nn.Linear(4, 1)#.to('cuda')
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = cpu_adam.DeepSpeedCPUAdam(model.parameters(), lr=0.001)
    
    for _ in range(3):
        input = torch.randn(4, 4)#.to('cuda')
        loss = model(input)
        loss = loss.sum()
        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
        torch.cuda.synchronize()
        
        for p in model.parameters():
            state = optimizer.state[p]
            print(id(p), id(state['exp_avg']), id(state['exp_avg_sq']))
            break

def cuda_trace_test():
    
    import torch.utils._cuda_trace as cuda_trace
    
    def callback(exp_id):
        def _callback(args):
            print('callback function fired')
            print(exp_id)
        return _callback
    
    linear = torch.nn.Linear(2,2, device='cuda')
    tensor = torch.randn(2,3, device='cuda')
    
    torch._C._activate_cuda_trace()
    cuda_trace.register_callback_for_cuda_memory_deallocation(callback(0))
    linear.to('cpu')
    #cuda_trace.pop_callback_for_cuda_memory_deallocation()
    tensor.to('cpu')
    print('trace test finished')

def queue_stream_test():
    stream = torch.cuda.Stream()
    stream.priority
    with torch.cuda.stream(stream):
        for i in range(5):
            b = torch.randn(2,10000).to('cuda')
    
    manager = torch.multiprocessing.Manager()
    queue = manager.Queue()
    queue.put(stream)
    
    def temp_func(queue):
        while not queue.query():
            continue
        print('enqueued job finished')
    p = Process(target=temp_func, args=(queue, ))
    p.start()
    p.join()
    
def cuda_enqueue_host_fn_test():
    from add_callback import add_callback
    #torch.cuda.Stream()
    
    input = torch.randn(200, 200).to('cuda')
    linear = torch.nn.Linear(200, 1)
    optim = torch.optim.Adam(linear.parameters(), lr=0.0001)
    linear.to('cuda')
    
    for param in linear.parameters():
        param._cpu_grad = torch.zeros_like(param.data, device="cpu").pin_memory()
    
    torch.cuda.synchronize()
    
    temp_stream = torch.cuda.Stream()
    print(temp_stream, temp_stream._cdata, type(temp_stream))
    
    def finalize_grads():
        for param in linear.parameters():
            param.grad = param._cpu_grad

    enqueue_host_fn = True #False
    
    # with torch.cuda.stream(temp_stream):
    output = linear(input)
    #print(output.shape)
    output.sum().backward()
    
    print(torch.cuda.current_stream())
    if enqueue_host_fn:
        add_callback(temp_stream.cuda_stream, finalize_grads, None)
    else:
        finalize_grads()
        
    print(torch.cuda.current_stream())
        
    optim.zero_grad()
    optim.step()

def h5_share_memory_test():
    ## Serialize Tensor
    random_tensor = torch.randn(512, 1024)
    torch.save(random_tensor, './outputs/random.pt')
    
    ## Convert to HDF5 Format
    h5_file_name = 'outputs/random.h5'

    if os.path.exists(h5_file_name):
        os.unlink(h5_file_name)

    with zipfile.ZipFile('outputs/random.pt', 'r') as zip_file:
        with h5py.File(h5_file_name, 'w') as h5_file:
            for info in zip_file.infolist():
                with zip_file.open(info.filename, 'r') as f:
                    file_data = f.read()
                    print(f'Copying {len(file_data)} bytes for "{info.filename}"')
                    dataset = h5_file.create_dataset(
                        info.filename, data=np.frombuffer(file_data, dtype=np.byte))

def print_model_id(rank, model):
        print(f'model process id: {id(model)}')
                         
def shared_memory_test():
    
    model = torch.nn.Linear(200, 1)
    print(f'model initialization: {id(model)}')
    model.share_memory()
    print(f'model share: {id(model)}')
    
    for param in model.parameters():
        print(type(param), type(param.data))
    ## NOTE
    ## torch.multiprocessing.Process works with shared memory
    ## but torch.multiprocessing.spawn is not.
    
    processes = []
    for rank in range(1):
        print(torch.multiprocessing.get_start_method())
        p = torch.multiprocessing.Process(target=print_model_id, args=(rank, model))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # torch.multiprocessing.spawn(
    #     fn=print_model_id,
    #     args=(model,),
    #     nprocs=1
    # )
    
    print(f'model share: {id(model)}')

from shared_memory import shared_memory
import nvtx
from cuda_enqueue import cuda_enqueue

def extended_shared_memory_test2(rank):
    
    with nvtx.annotate('slave tensor initalization'):
        tensor = torch.zeros(size=(50000, 100), device='cuda:0')
    print('Slave original', tensor, hex(tensor.data_ptr()))
    i = 0
    comm_stream = torch.cuda.Stream()
    comp_stream = torch.cuda.Stream()
    linear = torch.nn.ModuleList([
        torch.nn.Linear(100, 10000, device='cuda:0'),
        torch.nn.Linear(10000, 1, device='cuda:0'),
    ])
    output_list = []
    while i < 3:
        with nvtx.annotate('slave shared memory'):
            # with torch.cuda.stream(comm_stream):
            cuda_enqueue(tensor, 0, 0)
        with nvtx.annotate('slave forward'):
            # print('Slave After', tensor, hex(tensor.data_ptr()))
            with torch.cuda.stream(comp_stream):
                comp_stream.wait_stream(comm_stream)
                output = tensor
                for layer in linear:
                    output = layer(output)
            output_list.append(output.sum())
            # torch.cuda.synchronize()
        i += 1
    
    for output in output_list:
        print(output)
    # 

def extended_shared_memory_test():
    p = torch.multiprocessing.Process(target = extended_shared_memory_test2, args=(0,))
    p.start()
    
    with nvtx.annotate('master tensor initalization'):
        tensor = torch.arange(5000000, dtype=torch.float32).view(50000, 100).pin_memory()#torch.tensor([[0, 1, 2], [3, 4, 5]])#.to('cuda:0')
    
    i = 0
    while i < 3:
        with nvtx.annotate('master shared memory'):
            cuda_enqueue(tensor, 99, 0)
        
        with nvtx.annotate('master tensor change'):
            # tensor.copy_(torch.randn(10, 10))
            tensor[:,i] = 0
        i += 1
    #print('Master', tensor, error)
    p.join()

def d_to_d_copy_tensor_test():
    
    import storage_memory_utils
    
    size = (2048, 8192)
    cpu_tensor = torch.randn(size).pin_memory()
    gpu_tensor = torch.zeros(size, device='cuda')
    storage_memory_utils.free_storage_(gpu_tensor)
    torch.cuda.synchronize()
    
    with nvtx.annotate('GPU copy'):
        storage_memory_utils.alloc_storage_(gpu_tensor, size = gpu_tensor.size())
        gpu_tensor.copy_(cpu_tensor.to('cuda', non_blocking = True))
        torch.cuda.synchronize()
    
    storage_memory_utils.free_storage_(gpu_tensor)
    torch.cuda.synchronize()
    
    with nvtx.annotate('To & Copy'):
        storage_memory_utils.alloc_storage_(gpu_tensor, size = gpu_tensor.size())
        cpu_tensor = cpu_tensor.to('cuda', non_blocking=True)
        #torch.cuda.synchronize()
        gpu_tensor.copy_(cpu_tensor)

def d_to_d_copy_network_test():
    import storage_memory_utils
    
    size = (2048, 8192)
    linear = torch.nn.Linear(size[0], size[1])
    
    for param in linear.parameters():
        param._fp32 = param.data
        param._fp32 = param.pin_memory()
        param.data = param._fp32
        param._fp16 = torch.zeros_like(param._fp32, device = 'cuda:0')
        storage_memory_utils.free_storage_(param._fp16)
    
    torch.cuda.synchronize()
    
    with nvtx.annotate('GPU layer copy'):
        for param in linear.parameters():
            print(param)
            storage_memory_utils.alloc_storage_(param._fp16, size = param._fp16.size())
            param._fp16.copy_(
                param._fp32.to('cuda:0', non_blocking = True)
            )
            param.data = param._fp16

from shared_pinned_memory import shared_pinned_memory

def get_shared_pinned_params_test2():
    tensor = torch.zeros(500, 100)
    print('slave')
    new_tensor = shared_pinned_memory(tensor, 0, 0, 0, 0, 1)
    print('slave', new_tensor, new_tensor.storage().data_ptr())
    
    new_tensor[1] = 1
    print('slave new tensor modified', new_tensor, new_tensor.storage().data_ptr(), new_tensor.is_pinned())
    
def get_shared_pinned_params_test():
    p = torch.multiprocessing.Process(target=get_shared_pinned_params_test2, args=())
    p.start()
    tensor = torch.zeros(500, 100)
    
    new_tensor = shared_pinned_memory(tensor, 0, 0, 0, 0, 1)
    print('master', new_tensor, new_tensor.storage().data_ptr())
    
    p.join()
    print('master', new_tensor, new_tensor.storage().data_ptr(), new_tensor.is_pinned())

import argparse
from fairseq.modules.transformer_layer import FeedForwardNetwork
from time import sleep
from deepspeed.ops.adam import DeepSpeedCPUAdam
from utils import ParamType
from storage_memory_utils import see_memory_usage
@torch.no_grad()
def pre_define_optim_states(optim, rank, layer, num_experts, expert, num_orders):
    for group_id, group in enumerate(optim.param_groups):
        for param_id, p in enumerate(group['params']):
            state = optim.state[p]
            state['step'] = 0

            # gradient momentums
            state['exp_avg'] = shared_pinned_memory(p.data, rank, layer, num_experts, expert, num_orders, param_id, ParamType.OPTIM_EXP_AVG)
            
            #memory_format=torch.preserve_format)
            # gradient variances
            state['exp_avg_sq'] = shared_pinned_memory(p.data, rank, layer, num_experts, expert, num_orders, param_id, ParamType.OPTIM_EXP_AVG_SQ)
                
def expert_test(rank, args):
    embed_dim = args.decoder_embed_dim or args.encoder_embed_dim 
    ffn_dim = args.decoder_ffn_embed_dim or args.encoder_ffn_embed_dim
    
    expert = FeedForwardNetwork(args, embed_dim=embed_dim, ffn_dim = ffn_dim)
    
    optim = DeepSpeedCPUAdam(expert.parameters())

    see_memory_usage(rank)
    
    pre_define_optim_states(optim, rank, 0, 1, 0, 4)
    
    for id, param in enumerate(expert.parameters()):
        param._fp32 = param.data
        param._fp32 = shared_pinned_memory(param._fp32, rank, 0, 1, 0, 4, id, ParamType.PARAM)
        param.data = param._fp32
    
    if rank == 1:
        sleep(1)
    
    see_memory_usage(rank)
    
    print(f'Rank-{rank} Result')
    for p in expert.parameters():
        print(rank, p.data.norm())
    
def shared_pineed_network_test():
    parser = argparse.ArgumentParser()
    
    ## Training Args ##
    parser.add_argument('--embed-dim', default=2048, type=int)
    parser.add_argument('--seq-length', default=2048, type=int)
    parser.add_argument('--batch-size', default = 84, type=int)
    
    parser.add_argument('--expert-capacity', default=1.0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--moe-expert-count', default=4, type=int)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--opt-type', default='torch', type=str)
    
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
    
    torch.multiprocessing.spawn(
        fn=expert_test,
        args=(args, ),
        nprocs=2,
        join=True,
    )
    return

from utils import AverageMeter
import time
def norm_operation_speed_test():
    cpu_tensor = torch.randn(2048, 8192)
    gpu_tensor = torch.randn(2048, 8192, device='cuda:0')
    
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    
    cpu_meter = AverageMeter('cpu norm calculation')
    gpu_meter = AverageMeter('gpu norm calculation')
    
    for i in range(10):
        cpu_start = time.time() * 1000
        torch.isfinite(cpu_tensor).all()
        cpu_end = time.time() * 1000
        
        torch.cuda.synchronize()
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
        torch.isfinite(gpu_tensor).all()
        cuda_end.record()
        torch.cuda.synchronize()
        
        if i > 3:
            cpu_meter.update((cpu_end - cpu_start))
            gpu_meter.update(cuda_start.elapsed_time(cuda_end))
        
    print(cpu_meter, gpu_meter)
        
if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    #extended_shared_memory_test()
    #get_shared_pinned_params_test()
    # d_to_d_copy_network_test()
    #shared_pineed_network_test()
    norm_operation_speed_test()
    
    #queue_stream_test()
    #cuda_trace_test()
    #optimizer_states_address_test()
