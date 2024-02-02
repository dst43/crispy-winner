import torch
import torch.distributed as dist
import argparse
import os
import copy
import psutil
import time
from time import sleep
from fairseq.modules.transformer_layer import TransformerDecoderLayer
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()

parser.add_argument('--embed_dim', default=2048, type=int)
parser.add_argument('--ffn_dim', default=8192, type=int)

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
    
def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        init_method= 'env://', #'tcp://localhost:12355',
        world_size=world_size,
        rank=rank,
    )
    dist.barrier()

def main_fn(args):
    
    model_args = {'_name': 'transformer_lm_gpt', 'activation_fn': 'gelu', 'dropout': 0.1, 'attention_dropout': 0.1, \
        'activation_dropout': 0.0, 'relu_dropout': 0.0, 'decoder_embed_dim': 2048, 'decoder_output_dim': 2048,\
        'decoder_input_dim': 2048, 'decoder_ffn_embed_dim': 8192, 'decoder_layers': 1, 'decoder_attention_heads': 32,\
        'decoder_normalize_before': True, 'no_decoder_final_norm': False, 'adaptive_softmax_cutoff': None, \
        'adaptive_softmax_dropout': 0.0, 'adaptive_softmax_factor': 4.0, 'no_token_positional_embeddings': False, \
        'share_decoder_input_output_embed': True, 'character_embeddings': False, 'character_filters': '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]', \
        'character_embedding_dim': 4, 'char_embedder_highway_layers': 2, 'adaptive_input': False, \
        'adaptive_input_factor': 4.0, 'adaptive_input_cutoff': None, 'tie_adaptive_weights': False, \
        'tie_adaptive_proj': False, 'decoder_learned_pos': False, 'layernorm_embedding': False, \
        'no_scale_embedding': False, 'checkpoint_activations': True, 'offload_activations': False, \
        'decoder_layerdrop': 0.0, 'decoder_layers_to_keep': None, 'quant_noise_pq': 0.0, \
        'quant_noise_pq_block_size': 8, 'quant_noise_scalar': 0.0, 'min_params_to_wrap': 100000000, \
        'alternate_decoder_ffn_embed_dim': 0, 'moe_freq': 1, 'moe_expert_count': 4, 'moe_gating_use_fp32': True, \
        'moe_second_expert_policy': 'sampling', 'moe_normalize_gate_prob_before_dropping': False, 'moe_expert_ffn_dim': None, \
        'moe_top1_expert': True, 'moe_eval_capacity_token_fraction': -1.0, 'moe_normalize_expert_grad': 'sqrt_world_size', \
        'use_moe_pad_mask': False, 'record_a2a_perf_stats': False, 'dummy_a2a': False, 'moe_batch_prioritized_routing': False, \
        'use_stable_embedding': False, 'base_layers': 0, 'base_sublayers': 1, 'base_shuffle': 0, 'add_bos_token': False, \
        'tokens_per_sample': 2048, 'max_target_positions': 2048, 'tpu': False, 'memory_efficient_fp16': False, 'fp16': True, \
        'fp16_no_flatten_grads': True, 'ddp_backend': 'pytorch_ddp', 'world_size': 4, 'distributed_rank': 0, 'batch_size': 1, \
        'batch_size_valid': 1}
    model_args = argparse.Namespace(**model_args)
    model = TransformerDecoderLayer(model_args, no_encoder_attn=True, add_bias_kv=False, \
        add_zero_attn=False, is_moe_layer=True)
    
    ddp_model = DDP(model.to(args.rank), 
                    device_ids=[args.rank],
                    output_device=args.rank,
                    broadcast_buffers = False
                    )
    print(ddp_model)
    
    ddp_model.module.moe_layer.experts.to('cpu')
    torch.cuda.synchronize()
    print(f'rank: {args.rank}, expert_location: {next(ddp_model.module.moe_layer.experts.parameters()).device}')
    
    ddp_model.module.moe_layer.experts.to(f'cuda:{args.rank}')
    torch.cuda.synchronize()
    print(f'rank: {args.rank}, expert_location: {next(ddp_model.module.moe_layer.experts.parameters()).device}')
    
    input_dim = args.batch_size * args.seq_length * args.expert_capacity
    equal_a_to_a_cpu_meter = AverageMeter('equal_cpu_all_to_all')
    equal_a_to_a_gpu_meter = AverageMeter('equal_gpu_all_to_all')
    unequal_a_to_a_cpu_meter = AverageMeter('unequal_cpu_all_to_all')
    unequal_a_to_a_gpu_meter = AverageMeter('unequal_gpu_all_to_all')
    
    epoch = 10000
    
    for i in range(epoch):
        sleep(1)
    return

def all_to_all_wrapper(args, input, output, a_to_a_cpu_meter, a_to_a_gpu_meter, \
                                            input_list = None, output_list = None):
    # always record times, since it is not a lot of overhead
    # if we do not log it we simply clear it off in record_all_to_all_stats
    torch.cuda.synchronize()
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    cpu_start = time.time() * 1000
    cuda_start.record()
    dist.all_to_all_single(output, input, \
                            output_split_sizes=output_list, \
                            input_split_sizes=input_list, \
                            group=get_all2all_group(args.world_size))
    cuda_end.record()
    cpu_end = time.time() * 1000
    torch.cuda.synchronize()
    
    if a_to_a_cpu_meter:
        a_to_a_cpu_meter.update((cpu_end - cpu_start))
    
    if a_to_a_gpu_meter:
        a_to_a_gpu_meter.update(cuda_start.elapsed_time(cuda_end))
    
def distributed_comm(rank, world_size, main_fn, args):
    torch.cuda.set_device(rank)
    args.rank = rank
    args.world_size = world_size
    
    setup(rank, world_size)
    
    print(f'rank: {args.rank}, world_size: {args.world_size}')
    dist.all_reduce(torch.zeros(1).to(f'cuda:{args.rank}'))
    
    main_fn(args)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier(get_global_group())

def comm_test(args):
    args.batch_size = 4
    args.seq_length = 2048
    args.expert_capacity = 2
    args.embed_dim = 1024
    args.world_size = 4
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    torch.multiprocessing.spawn(
        fn=distributed_comm,
        args=(args.world_size, main_fn, args),
        nprocs=args.world_size,
        join=True,
    )
    
    return

if __name__ == "__main__":
    args = parser.parse_args()
    comm_test(args)