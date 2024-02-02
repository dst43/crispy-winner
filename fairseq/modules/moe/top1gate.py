# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

from typing import Callable, Dict, Tuple, Optional, List

import math
import torch
from torch import Tensor
import torch.nn.functional as F

from .moe_layer import has_tutel, fused_cumsum_sub_one
from .top2gate import one_hot, entropy

from torch.distributions import Categorical
from fairseq.distributed.utils import get_global_world_size, get_global_rank
# maximum capacity of 1 expert as a fraction of number of tokens in the batch
# Note: setting this to 1.0 causes inference to significantly slow down
EVAL_CAPACITY_TOKEN_FRACTION = 0.25

# logging
SAMPLE_FRACTION = 0.2

from fairseq import distributed_utils
from expertserver.utils import AverageMeter
import nvtx
import time

gate_all_gather_meter = None

def top1gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    capacity_factor=1.0,
    eval_mode=False,
    moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    gate_type = '',
    all2allgroup = None,
    experts = None,
    moe_cpu = False,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top2Gating on logits."""
    metadata = {}
    rank = get_global_rank()
    if use_fp32:
        orig_dtype = logits.dtype
        logits = logits.float()
    # print(f'logits: {logits}, {logits.shape}')
    gates = F.softmax(logits, dim=1)
    # print(f'gates: {gates}, {gates.shape}')

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
        # print(f'capacity: {capacity}')
    # Create a mask for 1st's expert per token
    indices1_s = torch.argmax(gates, dim=1)
    # print(f'indices1_s: {indices1_s}, {indices1_s.shape}')
    mask1 = one_hot(indices1_s, num_classes=num_experts, unsqueeze_indices=True)
    # print(f'mask1: {mask1}, {mask1.shape}')
    
    if input_mask is not None and input_mask.any():
        nonpadding = ~ input_mask
        mask1 = mask1 * nonpadding.unsqueeze(-1).to(mask1.dtype)

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.to(gates.dtype), dim=0)
    # print(f'gates: {gates}, {gates.shape}')
    # print(f'me: {me}, {me.shape}')
    # print(f'mask1: {mask1}, {mask1.shape}')
    # print(f'ce: {ce}, {ce.shape}')
    # print(f'num_experts: {num_experts}')
    l_aux = torch.mean(me * ce)
    l_aux = l_aux * num_experts * num_experts
    
    expert1_hist = 100 * torch.histc((indices1_s.squeeze() + 1), bins=num_experts, min=1, max=num_experts) / num_tokens
    metadata["unused_expert1_count"] = (expert1_hist == 0).sum()
    expert1_hist = torch.sort(expert1_hist, dim=0, descending=True).values + torch.finfo(torch.float32).tiny

    sample_count = max(math.ceil(num_experts * SAMPLE_FRACTION), 1)
    metadata["expert1_balance_top"] = expert1_hist[:sample_count].sum()
    metadata["expert1_balance_bottom"] = expert1_hist[-sample_count:].sum()

    ############# Baseline? ########
    ############# All Reduce to get the maximum capacity ###################
    all2all_size = distributed_utils.get_world_size(all2allgroup)
    
    # cuda_start = torch.cuda.Event(enable_timing=True)
    # cuda_end = torch.cuda.Event(enable_timing=True)       
    # # torch.distributed.barrier()
    # torch.cuda.synchronize()
    # cuda_start.record()
    # global gate_all_gather_meter
    
    if gate_type == 'baseline':
        gatestats = mask1.sum(dim=0)
        # print(f'gatestats: {gatestats}, {gatestats.shape}')
        if all2all_size > 1:
            torch.distributed.all_reduce(gatestats, group = all2allgroup, op = torch.distributed.ReduceOp.MAX)
        capacity = int(torch.max(gatestats).item())
        # print(f'cap: {capacity}')
        gpu_expert_assign, gather_list = [], []
        # cuda_end.record()
    ############# Ours ########
    ############# All Gather all gatestats ############
    elif gate_type == 'ours':
        gatestats = mask1.sum(dim=0)
        gather_list = [torch.zeros_like(gatestats) for _ in range(all2all_size)]
        
        ## Call all previous optimizationo sync. before all gather to sync. with other GPUs.
        if moe_cpu:
            cpu_start = time.time() * 1000 
            with nvtx.annotate('wait for prev optim'):
                experts._wait_for_previous_optim_step()
            cpu_end = time.time() * 1000
            # print(f'{(cpu_end - cpu_start)} ms took to join all threads')
        if all2all_size > 1:
            torch.distributed.all_gather(gather_list, gatestats, group = all2allgroup)
        else:
            gather_list[0] = gatestats
        capacity = 0
        for stat in gather_list:
            capacity = max(capacity, int(stat.max().item()))
        #capacity = int(max(gather_list, key=lambda tensor: int(tensor.max().item())).max().item())#int(summed_list.max().item())
    
        summed_list = torch.stack(gather_list, dim = 0).sum(dim = 0)    
        sorted_list, indices = summed_list.sort(descending = True)
        sorted_list, indices = sorted_list.tolist(), indices.tolist()
        bins, gpu_expert_assign = [(0, 0) for _ in range(all2all_size)], [list() for _ in range(all2all_size)]
        max_assign = num_experts // all2all_size
        for i in range(len(sorted_list)):
            idx_min = min(range(len(bins)), key=lambda idx: bins[idx][0] if bins[idx][1] < max_assign else float('inf'))
            val, count = bins[idx_min]
            bins[idx_min] = (val + sorted_list[i], count + 1)
            gpu_expert_assign[idx_min].append(indices[i])
        
        main_stream = torch.cuda.current_stream()
        
        if moe_cpu:
            # first_comm_stream = experts._streams['communication']
            # first_comm_stream.wait_stream(main_stream)
            # first_exp_id = gpu_expert_assign[rank][0]
            # experts._upload_params(experts.expert_list[first_exp_id].parameters())
            for idx, exp_id in enumerate(gpu_expert_assign[rank]):
                comm_stream = experts.comm_streams[exp_id]
                if idx == 0:
                    comm_stream.wait_stream(main_stream)
                else:
                    comm_stream.wait_stream(experts.comm_streams[prev_exp_id])
                prev_exp_id = exp_id
                with torch.cuda.stream(comm_stream):
                    experts._upload_params(experts.expert_list[exp_id].parameters())
                    
        # print(gather_list, gpu_expert_assign)
        gpu_expert_assign_flatten = [item for sublist in gpu_expert_assign for item in sublist]
        # print(gatestats, gatestats[gpu_expert_assign_flatten])
        
        counts = torch.cumsum(gatestats[gpu_expert_assign_flatten], dim=0)
        counts = torch.cat([torch.tensor([0], dtype=counts.dtype, device='cuda'), counts[:-1]])
        # print(counts)
        tensor_list = []
        for i, exp_id in enumerate(gpu_expert_assign_flatten):
            tensor = indices1_s == exp_id
            tensor = tensor.to(indices1_s.dtype) * counts[i]
            # print(tensor)
            tensor_list.append(tensor)
        
        temp = torch.stack(tensor_list).sum(dim=0)
        # print(temp)
        gates = mask1 * gates
        gates1_s = (gates * mask1).sum(dim=1)
        # print(f'gate1_s: {gates1_s}, {gates1_s.shape}')
        # Compute locations in capacity buffer
        locations1 = fused_cumsum_sub_one(mask1)
        # print(f'locations1: {locations1}, {locations1.shape}')
        # print(f'mask1: {mask1}, {mask1.shape}')
        # Store the capacity location for each token
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        # print(f'locations1 * mask: {locations1 * mask1}, {mask1.shape}')
        # print(f'locations1_s: {locations1_s}, {locations1_s.shape}')
        # print(f'final: {locations1_s + temp}')
        index_list = locations1_s + temp
        # print(f'final: {temp + indices1_s}')
        # assert False
        # counts = fused_cumsum_sub_one(gatestats[gpu_expert_assign_flatten])
        # print(f'counts: {counts}, {counts.shape}')
        # # counts = counts[list(reversed(gpu_expert_assign_flatten))]
        # # print(f'counts_back: {counts}, {counts.shape}')
        # cuda_end.record()
        # index_list = []
        # print(f'reversed indices1_s: {reversed(indices1_s)}')
        # for idx in reversed(indices1_s):
        #     index_list.append(counts[idx].item())
        #     counts[idx] -= 1
        # index_list.reverse()
        
        
        # print(f'index_list: {index_list}')
        

        # assert False
        ## TODO
        ## Change it to index_list.clone().deteach().to('cuda')
        my_matrix = one_hot(torch.tensor(index_list, device='cuda'), num_classes=num_tokens, unsqueeze_indices=True)
        # # print(f'my matrix: {my_matrix}')
    
        
        gates_flatten = gates.flatten()
        non_zero_mask = (gates_flatten != 0)
        gates_values = torch.masked_select(gates_flatten, non_zero_mask)
        # print(gates_values)
        gates_values = gates_values[:my_matrix.shape[0]]
        combine1_sec = my_matrix * gates_values.unsqueeze(dim=-1)
        # print(combine1_sec)
        # cuda_end.record()
        dispatch_mask = combine1_sec.bool()
        # print(dispatch_mask)
        if use_fp32:
            combine1_sec = combine1_sec.to(orig_dtype)
        
        
        # torch.cuda.synchronize()
        # gather_process.update(cuda_start.elapsed_time(cuda_end))
        # if not gate_all_gather_meter:
        #     gate_all_gather_meter = AverageMeter('gate all gather meter')
        # gate_all_gather_meter.update(cuda_start.elapsed_time(cuda_end))
        # print(f'{get_global_rank()}, count {gate_all_gather_meter.count}, {gate_all_gather_meter}')

        return l_aux, combine1_sec, dispatch_mask, metadata, gpu_expert_assign, gather_list
    
    elif gate_type == 'drop':
        if moe_eval_capacity_token_fraction > 0.0 and eval_mode:
            capacity = math.ceil(moe_eval_capacity_token_fraction * num_tokens)
        else:
            # capacity = capacity_factor * S/E
            capacity = int(capacity_factor * math.ceil(num_tokens / num_experts))
        gpu_expert_assign, gather_list = [], []
    
        gatestats = mask1.sum(dim=0)
        if all2all_size > 1:
            torch.distributed.all_reduce(gatestats, group = all2allgroup, op = torch.distributed.ReduceOp.MAX)
        temp = int(torch.max(gatestats).item())
        # print(f'max_num_tokens: {temp}')
    else:
        assert False, "NO GATE TYPES ARE GIVEN"
    
    # torch.cuda.synchronize()
    # # gather_process.update(cuda_start.elapsed_time(cuda_end))
    # global gate_all_gather_meter
    # if not gate_all_gather_meter:
    #     gate_all_gather_meter = AverageMeter('gate all gather meter')
    # gate_all_gather_meter.update(cuda_start.elapsed_time(cuda_end))
    # print(f'{get_global_rank()}, count {gate_all_gather_meter.count}, {gate_all_gather_meter}')
    # metadata["extra_padding"] = (capacity - int(capacity_factor * math.ceil(num_tokens / num_experts))) * num_experts
    
    #print(f'cap: {capacity} vs {int(capacity_factor * math.ceil(num_tokens / num_experts))}')
    ######################################### GATE STATS ########################################
    # each_gpu_num_experts = num_experts // get_global_world_size()
    # gatestats = mask1.sum(dim=0)#.clamp(max = each_gpu_num_experts * capacity).detach()
    # print(gatestats)
    # gatestats_probs = gatestats / num_tokens
    # etntropy_gatestats = Categorical(probs = gatestats_probs).entropy()
    # metadata["etntropy_gatestats"] = etntropy_gatestats
    
    # gatestats_probs, indices = gatestats_probs.sort(dim=0, descending=True)
    # print(gatestats, indices)
    # gatestats_probs[[i for i in range(each_gpu_num_experts)]] = 0
    # metadata["communication_room"] = abs(gatestats_probs.sum(dim=0) - (1 - 1 / get_global_world_size()))
    
    # ### This assumes multiple experts per GPU
    # each_gpu_num_experts = num_experts // num_experts #get_global_world_size()
    # gatestats_list = []
    # for i in range(0, list(gatestats.shape)[0], each_gpu_num_experts):
    #     gatestats_list.append(gatestats[i : i + each_gpu_num_experts].sum(dim=0, keepdim=True).clamp(max = each_gpu_num_experts * capacity))
    # gatestats_GPU, indices = torch.cat(gatestats_list, dim = 0).sort(dim=0, descending=True)
    # gatestats_GPU_probs =  gatestats_GPU / num_tokens
    # # metadata["communication_room"] =  abs(gatestats_GPU_probs.sum() - (1 - 1 / num_experts))
    
    # ### This assumes single expert per GPU
    # each_gpu_num_experts = num_experts // get_global_world_size()
    # #print(gatestats_GPU, get_global_rank() * each_gpu_num_experts, (get_global_rank() + 1) * each_gpu_num_experts)
    # comm_stats = 0
    # for i in range(each_gpu_num_experts):
    #     temp = gatestats_GPU_probs[i]
    #     gatestats_GPU_probs[i] = 0
    #     comm_stats += abs(gatestats_GPU_probs.sum() - (1 - 1 / num_experts))
    #     gatestats_GPU_probs[i] = temp
    
    # for i in range(get_global_rank() * each_gpu_num_experts, (get_global_rank() + 1) * each_gpu_num_experts):
    #     temp = gatestats_GPU_probs[i]
    #     gatestats_GPU_probs[i] = 0
    #     comm_stats += abs(gatestats_GPU_probs.sum() - (1 - 1 / num_experts))
    #     gatestats_GPU_probs[i] = temp
    
    # metadata["communication_room"] = comm_stats / each_gpu_num_experts
    #############################################################################################
    # print(f'capacity" {capacity}')
    # print(f'expert_statistics: {stats}, {stats.shape}')
    # print(f'expert_stats_probs: {stats_probs}, {stats_probs.shape}')
    #print(f'etntropy_gatestats: {gate_stat_entropy}')
    
    # print(mask1, mask1.shape, mask1.sum(dim=0))

    # for logging (percent of tokens routed to each expert)

    gates1_s = (gates * mask1).sum(dim=1)
    # print(f'gate1_s: {gates1_s}, {gates1_s.shape}')
    # Compute locations in capacity buffer
    locations1 = fused_cumsum_sub_one(mask1)
    # print(f'locations1: {locations1}, {locations1.shape}')

    # assert False
    # print(f'l_aux: {l_aux}')


    if has_tutel:
        locations1_s = torch.sum(locations1 * mask1, dim=1)
        return l_aux, metadata, capacity, num_experts, [indices1_s,], [locations1_s,], [gates1_s,]

    # Remove locations outside capacity from mask
    if gate_type not in ('basline', 'ours'):
        mask1 = mask1 * torch.lt(locations1, capacity)
    # print(f'mask1: {mask1}, {mask1.shape}')
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    # print(f'locations1 * mask: {locations1 * mask1}, {mask1.shape}')
    # print(f'locations1_s: {locations1_s}, {locations1_s.shape}')
    # print(f'final: {locations1_s + temp}')

    # Calculate combine_weights and dispatch_mask
    gates1 = gates1_s.unsqueeze(-1) * mask1.to(gates1_s.dtype)  # einsum("s,se->se")
    # print(f'gates1: {gates1}, {gates1.shape}')
    # locations1_sc = num_tokens * capacity
    locations1_sc = one_hot(locations1_s, num_classes=capacity, unsqueeze_indices=True)
    # print(f'locations1_sc: {locations1_sc}, {locations1_sc.shape}')
    combine1_sec = torch.bmm(
        # einsum("se,sc->sec")
        gates1.unsqueeze(-1), locations1_sc.to(gates1.dtype).unsqueeze(1)
    )
    # print(f'combine1_sec: {combine1_sec}, {combine1_sec.shape}')
    dispatch_mask = combine1_sec.bool()
    # print(f'dispatch_mask: {dispatch_mask}, {dispatch_mask.shape}')
    # print(dispatch_mask.shape)
    # 
    if use_fp32:
        return l_aux, combine1_sec.to(orig_dtype), dispatch_mask, metadata, gpu_expert_assign, gather_list
    else:
        return l_aux, combine1_sec, dispatch_mask, metadata, gpu_expert_assign, gather_list


class Top1Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        input_noise_type=None,
        capacity_factor=1.0,
        moe_eval_capacity_token_fraction=EVAL_CAPACITY_TOKEN_FRACTION,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()
        self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        self.use_fp32 = use_fp32
        self.input_noise_type = input_noise_type
        self.capacity_factor = capacity_factor
        self.moe_eval_capacity_token_fraction = moe_eval_capacity_token_fraction

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None, \
                gate_type = '', all2allgroup = None, experts = None, moe_cpu = False) -> Tuple[Tensor, Tensor, Tensor, Dict]:  # type: ignore
        logits = self.wg(input)
        # print(f'input: {torch.norm(input, p=2, dtype=torch.float32)}')
        # for param in self.wg.parameters():
        #     print(f'gate params: {torch.norm(param, p=2, dtype=torch.float32)}')
        return top1gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            capacity_factor=self.capacity_factor,
            eval_mode=not self.training,
            moe_eval_capacity_token_fraction=self.moe_eval_capacity_token_fraction,
            gate_type = gate_type,
            all2allgroup = all2allgroup,
            experts = experts,
            moe_cpu = moe_cpu
        )
