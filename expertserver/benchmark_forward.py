import nvtx
import torch
import argparse
import time
import gc
import psutil

from fairseq import optim
from fairseq.modules.transformer_layer import FeedForwardNetwork
from omegaconf import OmegaConf

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

def module_to_test(args):
    args.seq_length = 2048
    args.batch_size = 4
    args.dropout = 0.1
    args.pin_memory = True
    
    expert = FeedForwardNetwork(args, embed_dim=args.embed_dim, ffn_dim = args.ffn_dim)
    
    print("__named_buffers__")
    for b in expert.named_buffers():
        print(b)
    
    print("__named_children__")
    for c in expert.named_children():
        print(c)
    
    print("__named_modules__")
    for m in expert.named_modules():
        print(m)
    
    print("__named_parameters__")
    for p in expert.named_parameters():
        print(p)

    print(expert.state_dict)
    
    print(expert.to_empty(device='cuda'))

def see_memory_usage(rank = 0):

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    print(
        f"rank: {rank},  MA {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        Max_MA {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB")

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    print(
        f'rank: {rank}, CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()

def optimizer_to(optim, device, non_blocking):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device, non_blocking=non_blocking)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device, non_blocking=non_blocking)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device, non_blocking=non_blocking)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device, non_blocking=non_blocking)

def fairseq_optimizer_to(optim, device, non_blocking):
    # for p16 in optim.fp16_params:
    #     p16.data = p16.data.to(device, non_blocking=non_blocking)
    #     if p16._grad is not None:
    #         p16._grad.data = p16._grad.data.to(device, non_blocking=non_blocking)
    for p32 in optim.fp32_params.values():
        p32.data = p32.data.to(device, non_blocking=non_blocking)
        if p32._grad is not None:
            p32._grad.data = p32._grad.data.to(device, non_blocking=non_blocking)
    optimizer_to(optim.fp32_optimizer._optimizer, device, non_blocking)

def main(args):
    
    input_dim = int(args.batch_size * args.seq_length * args.expert_capacity)
    dtype = torch.float16 if args.fp16 else torch.float32
    dummy_input = torch.randn(input_dim, args.embed_dim, dtype=dtype).to('cuda')
    expert = FeedForwardNetwork(args, embed_dim=args.embed_dim, ffn_dim = args.ffn_dim)
    
    print(expert)
    num_of_elements = sum(p.numel() for p in expert.parameters() if p.requires_grad)
    print(num_of_elements)
    print(f"Expert Size : {num_of_elements * 20 / 1000000}MB")
    
    if args.fp16:
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
        expert.half()
        optimizer = optim.FP16Optimizer.build_optimizer(cfg_dls, list(expert.parameters()))
        expert.to('cuda', non_blocking=True)
        fairseq_optimizer_to(optimizer, 'cuda', non_blocking=True)
    elif args.amp:
        scaler = torch.cuda.amp.GradScaler()
        optimizer = optim.Adam(expert.parameters(), lr = 0.0001)
        expert.to('cuda', non_blocking=True)
        optimizer_to(optimizer, 'cuda', non_blocking=True)
    else:
        ### Create Experts first and send to CUDA to 
        ### make sure the created optimizer states are pinned on CPU
        
        optimizer = torch.optim.Adam(expert.parameters(), lr = 0.0001)
        expert.to('cuda', non_blocking=True)
        optimizer_to(optimizer, 'cuda', non_blocking=True)
    
    ## Single Optimization Step to make optimizer states
    optimizer.zero_grad()

    if args.fp16:
        output = expert(dummy_input)
        loss = output.mean()
        optimizer.backward(loss)
        optimizer.step()
    elif args.amp:
        dummy_input = dummy_input.to('cuda')
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = expert(dummy_input)
        loss = output.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        expert.to('cpu', non_blocking=True)
        optimizer_to(optimizer, 'cpu', non_blocking=True)
        del dummy_input
    else:
        output = expert(dummy_input)
        loss = output.mean()
        loss.backward()
        optimizer.step()
    
    see_memory_usage()
    
    params_cpu_to_gpu = AverageMeter('params_cpu_to_gpu')
    params_gpu_to_cpu = AverageMeter('params_gpu_to_cpu')
    optim_cpu_to_gpu = AverageMeter('optim_cpu_to_gpu')
    optim_gpu_to_cpu = AverageMeter('optim_gpu_to_cpu')
    forward_meter = AverageMeter('forward')
    backward_meter = AverageMeter('backward')
    optimize_meter = AverageMeter('optimize')
    
    for i in range(args.epochs):
        if i < 3:
            ### TO CPU ###
            expert.to('cpu', non_blocking=True)
            for param in expert.parameters():
                dev = param.device
                print(dev)
                break
            if args.fp16:
                fairseq_optimizer_to(optimizer, 'cpu', non_blocking=True)
                for p32 in optimizer.fp32_params.values():
                    print(f'fp32 params device: {p32.device}')
                    break
                for param in optimizer.fp32_optimizer._optimizer.state.values():
                    if isinstance(param, torch.Tensor):
                        print(f'optim device: {param.device}')
                        break
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                print(f'optim device: {subparam.device}')
                            break
                        break
            else:
                optimizer_to(optimizer, 'cpu', non_blocking=True)
                for param in optimizer.state.values():
                    if isinstance(param, torch.Tensor):
                        print(f'optim device: {param.device}')
                        break
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                print(f'optim device: {subparam.device}')
                            break
                        break
            
            ### TO CUDA ###
            expert.to('cuda', non_blocking=True)
            for param in expert.parameters():
                dev = param.device
                print(dev)
                break
            
            if args.fp16:
                fairseq_optimizer_to(optimizer, 'cuda', non_blocking=True)
                for p32 in optimizer.fp32_params.values():
                    print(f'fp32 params device: {p32.device}')
                    break
                for param in optimizer.fp32_optimizer._optimizer.state.values():
                    if isinstance(param, torch.Tensor):
                        print(f'optim device: {param.device}')
                        break
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                print(f'optim device: {subparam.device}')
                            break
                        break
            else:
                optimizer_to(optimizer, 'cuda', non_blocking=True)
                for param in optimizer.state.values():
                    if isinstance(param, torch.Tensor):
                        print(f'optim device: {param.device}')
                        break
                    elif isinstance(param, dict):
                        for subparam in param.values():
                            if isinstance(subparam, torch.Tensor):
                                print(f'optim device: {subparam.device}')
                            break
                        break
        else:
            ### TO CPU ###
            
            ### Parmas ###
            stream = torch.cuda.current_stream()
            stream.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            expert.to('cpu', non_blocking=True)
            cuda_end.record()
            stream.synchronize()
            params_cpu_to_gpu.update(cuda_start.elapsed_time(cuda_end))
            
            ### Optim States ###
            stream = torch.cuda.current_stream()
            stream.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            if args.fp16:
                fairseq_optimizer_to(optimizer, 'cpu', non_blocking=True)
            else:
                optimizer_to(optimizer, 'cpu', non_blocking=True)
            cuda_end.record()
            stream.synchronize()
            optim_cpu_to_gpu.update(cuda_start.elapsed_time(cuda_end))
            
            ### TO CUDA ###
            stream = torch.cuda.current_stream()
            stream.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            expert.to('cuda', non_blocking=True)
            cuda_end.record()
            stream.synchronize()
            params_gpu_to_cpu.update(cuda_start.elapsed_time(cuda_end))
            
            stream = torch.cuda.current_stream()
            stream.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            if args.fp16:
                fairseq_optimizer_to(optimizer, 'cuda', non_blocking=True)
            else:
                optimizer_to(optimizer, 'cuda', non_blocking=True)
            cuda_end.record()
            stream.synchronize()
            optim_gpu_to_cpu.update(cuda_start.elapsed_time(cuda_end))
            
            # ### TO CPU ###
            # with nvtx.annotate(f"G_to_C{i-2}", color="orange"):
            #     if args.fp16:
            #         fairseq_optimizer_to(optimizer, 'cpu', non_blocking=True, optimizer=False)
            #     else:
            #         expert.to('cpu', non_blocking=True)
            #         optimizer_to(optimizer, 'cpu', non_blocking=True)
            #     torch.cuda.synchronize()
            
            # ### TO CUDA ###
            # with nvtx.annotate(f"C_to_G{i-2}", color="red"):
            #     if args.fp16:
            #         fairseq_optimizer_to(optimizer, 'cuda', non_blocking=True, optimizer=False)
            #     else:
            #         expert.to('cuda', non_blocking=True)
            #         optimizer_to(optimizer, 'cuda', non_blocking=True)
            #     torch.cuda.synchronize()
                
            # if i == 5:
            #     for p16 in optimizer.fp16_params:
            #         print(f'p16: {p16.data.is_pinned(), p16._grad.data.is_pinned()}')
                    
            #     for p32 in optimizer.fp32_params.values():
            #         print(f'p32: {p32.data.is_pinned()}, {p32._grad.data.is_pinned()}')
                
            #     for param in optimizer.fp32_optimizer._optimizer.state.values():
            #         for subparam in param.values():
            #             if isinstance(subparam, torch.Tensor):
            #                 print(f'optim: {subparam.data.is_pinned()}')
            #                 if subparam._grad is not None:
            #                     print(f'grad: {subparam._grad.is_pinned()}')
    
    see_memory_usage()
    # for p16 in optimizer.fp16_params:
    #     print(f'p16: {p16.data.device, p16._grad.data.device}')
        
    # for p32 in optimizer.fp32_params.values():
    #     print(f'p32: {p32.data.device}, {p32._grad.data.device}')
    
    # for param in optimizer.fp32_optimizer._optimizer.state.values():
    #     for subparam in param.values():
    #         if isinstance(subparam, torch.Tensor):
    #             print(f'optim: {subparam.data.device}')
    #             if subparam._grad is not None:
    #                 print(f'grad: {subparam._grad.device}')
    
    for i in range(args.epochs):
        optimizer.zero_grad()
        if i < 3:
            if args.amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = expert(dummy_input)
                    loss = output.mean()
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif args.fp16:
                output = expert(dummy_input)
                loss = output.mean()
                see_memory_usage()
                optimizer.backward(loss)
                optimizer.step()
            else:
                output = expert(dummy_input)
                loss = output.mean()
                loss.backward()
                optimizer.step()
        else:
            stream = torch.cuda.current_stream()
            stream.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            if args.amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = expert(dummy_input)
            else:
                output = expert(dummy_input)
            cuda_end.record()
            stream.synchronize()
            forward_meter.update(cuda_start.elapsed_time(cuda_end))
            
            stream = torch.cuda.current_stream()
            stream.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            loss = output.mean()
            if args.amp:
                scaler.scale(loss).backward()
            elif args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            cuda_end.record()
            stream.synchronize()
            backward_meter.update(cuda_start.elapsed_time(cuda_end))
            
            stream = torch.cuda.current_stream()
            stream.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            cuda_end.record()
            stream.synchronize()
            optimize_meter.update(cuda_start.elapsed_time(cuda_end))
                
            # torch.cuda.synchronize()
            # with nvtx.annotate(f"Forward{i-2}", color="purple"):
            #     if args.amp:
            #         with torch.cuda.amp.autocast(dtype=torch.float16):
            #             output = expert(dummy_input)
            #     else:
            #         output = expert(dummy_input)
            #     torch.cuda.synchronize()
            
            # with nvtx.annotate(f"Backward{i-2}", color="blue"):
            #     loss = output.mean()
            #     if args.amp:
            #         scaler.scale(loss).backward()
            #     elif args.fp16:
            #         optimizer.backward(loss)
            #     else:
            #         loss.backward()
            #     torch.cuda.synchronize()

            # with nvtx.annotate(f"Optimize{i-2}", color="magenta"):
            #     if args.amp:
            #         scaler.step(optimizer)
            #         scaler.update()
            #     else:
            #         optimizer.step()
            #     torch.cuda.synchronize()
    print(f'Expert Size: ({args.embed_dim}, {args.ffn_dim}), Tensor Size: ({input_dim}, {args.embed_dim}), FP16: {args.fp16}')
    print(params_cpu_to_gpu)
    print(params_gpu_to_cpu)
    print(optim_cpu_to_gpu)
    print(optim_gpu_to_cpu)
    print(forward_meter)
    print(backward_meter)
    print(optimize_meter)
    return

def memcpy_rapper(model, device, cpu_meter, gpu_meter):
    stream = torch.cuda.current_stream()
    stream.synchronize()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.time() * 1000
    cuda_start.record()
    model.to(device, non_blocking=True)
    cuda_end.record()
    cpu_end = time.time() * 1000
    cpu_meter.update(cpu_end - cpu_start)
    stream.synchronize()
    gpu_meter.update(cuda_start.elapsed_time(cuda_end))
    return

def forward_rapper(model, input, cpu_meter, gpu_meter):
    stream = torch.cuda.current_stream()
    stream.synchronize()
    
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.time() * 1000
    cuda_start.record()
    output = model(input)
    cuda_end.record()
    cpu_end = time.time() * 1000
    cpu_meter.update(cpu_end - cpu_start)
    
    stream.synchronize()
    gpu_meter.update(cuda_start.elapsed_time(cuda_end))
    return output

def backward_rapper(loss, cpu_meter, gpu_meter):
    
    stream = torch.cuda.current_stream()
    stream.synchronize()
    
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.time() * 1000
    cuda_start.record()
    loss.backward()
    cuda_end.record()
    cpu_end = time.time() * 1000
    cpu_meter.update(cpu_end - cpu_start)
    stream.synchronize()
    gpu_meter.update(cuda_start.elapsed_time(cuda_end))
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--embed-dim', default=1024, type=int)
    #parser.add_argument('--ffn-dim', default=4096, type=int)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--seq-length', default=2048, type=int)
    parser.add_argument('--expert-capacity', default=1.0, type=float)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    args = parser.parse_args()
    args.ffn_dim = args.embed_dim * 4
    main(args)