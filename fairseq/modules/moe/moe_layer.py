# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast, List

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda import Event as CudaEvent
from torch.nn import Module, ModuleList
from fairseq import distributed_utils
import builtins
import time
import copy
import numpy as np
import nvtx

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    has_tutel, fused_cumsum_sub_one = True, tutel_moe.fast_cumsum_sub_one
except ModuleNotFoundError:
    has_tutel, fused_cumsum_sub_one = False, lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)

from .top2gate import one_hot
# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor, input_splits = None, output_splits = None) -> Tensor:  # type: ignore
        # print(f'group: {group}')
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        
        input = input.contiguous()
        # print(f'{distributed_utils.get_global_rank()} all2all input: {input.norm()}')
        ## output shape should also be dynamically change
        if input_splits and output_splits:
            output = torch.zeros((sum(output_splits), input.shape[1]), dtype=input.dtype, device=input.device)
        else:
            output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, output_split_sizes=output_splits, input_split_sizes=input_splits, group=group)
        else:
            assert group is None
            output = input
        # print(f'{distributed_utils.get_global_rank()} after output: {output.norm()}')
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output, ctx.output_splits, ctx.input_splits), None, None)


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
        if self.count < 5:
            return
        self.val = val
        self.sum += val * n
        self.avg = self.sum / (self.count - 4)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList, List], args, group: Optional[Any] = None, all2all_group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        # if type(experts) == ModuleList:
        #     print('modulelist')
        #     self.experts = cast(ModuleList, experts)
        # ## For CPU MoE
        # # elif type(experts) == Module:
        # #     print('module')
        # #     self.experts = experts
        # else:
        #     #print('wtf?', type(experts))
        self.experts = experts #ModuleList([experts])
        
        self.expert_group = group if group is not None else distributed_utils.get_moe_group(args.moe_expert_count)
        self.all2all_group = all2all_group if all2all_group is not None else distributed_utils.get_all2all_group(args.moe_expert_count)
        
        if not args.moe_cpu:
            for p in experts.parameters():
                p.expert = True  # type: ignore
                
        self.world_size = distributed_utils.get_world_size(self.expert_group)
        self.all2all_size = distributed_utils.get_world_size(self.all2all_group)
        # if not args.moe_cpu:
        # self.num_local_experts = len(self.experts)
        # else:
        self.num_local_experts = args.moe_expert_count // (1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size())
        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0
        
        self.cpu_gate_meter = AverageMeter('cpu_gate_meter')
        self.gpu_gate_meter = AverageMeter('gpu_gate_meter')
        
        self.cpu_alltoall_meter = AverageMeter('cpu_alltoall_meter')
        self.gpu_alltoall_meter = AverageMeter('gpu_alltoall_meter')
        
        self.cpu_expert_meter = AverageMeter('cpu_expert_meter')
        self.gpu_expert_meter = AverageMeter('gpu_expert_meter')
        
        self.gpu_postprocess_meter = AverageMeter('gpu_input_postprocess_meter')
        
        self.token_drop_meter = AverageMeter('token drop meter')
        self.baseline_meter = AverageMeter('baseline meter')
        self.baseline_nonzero_meter = AverageMeter('baseline nonzero meter')
        self.ours_meter = AverageMeter('ours meter')
        self.rank = distributed_utils.get_global_rank()
        

    def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:
        
        if self.rank != 0:
            def print_pass(*args, **kwargs):
                pass
            builtins.print = print_pass
                
        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        if input_padding_mask is not None:
            assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)
        expected_bsz = getattr(self.args, 'batch_size', 0) if self.training else getattr(self.args, 'batch_size_valid', 0)
        # This indicates that --batch-size or --max-sentences is not specified
        if expected_bsz is None:
            expected_bsz = 0
        # Note: Padding is not necessary at generation time at present
        # because all DDP workers process the same batch. Also, batch size at generation time
        # can be different from that present in the checkpoint state
        # print(f'input_shape: {input_shape}')
        if not self.in_generation and expected_bsz != 0 and input_shape[0] != expected_bsz:
            logger.warning(f"padding batch with unexpected size {input_shape[0]} (expected: {expected_bsz})")
            assert input_shape[0] < expected_bsz, f"{input_shape[0]} < {expected_bsz}"
            padded_input = torch.zeros(
                (expected_bsz, input_shape[1], input_shape[2]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:input_shape[0], :, :] = input
            input = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_bsz, input_shape[1], ), dtype=torch.bool, device=input.device
            )
            if input_padding_mask is not None:
                padded_input_padding_mask[:input_shape[0], :] = input_padding_mask
            else:
                padded_input_padding_mask[:input_shape[0], :] = False
            input_padding_mask = padded_input_padding_mask
        # print(f'input.shape: {input.shape}')
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        #print(f'reshaped_input_shape : {reshaped_input_shape}')
        reshaped_input_padding_mask = input_padding_mask.reshape(-1) if input_padding_mask is not None else None

        # Doing padding here when --max-tokens is specified and not --batch-size or --max-sentences
        # Pro of --max-tokens: more flexible for MT variable sequence lengths
        # Con of --max-tokens: extra all-reduce needed to figure out optimal padding without running OOM
        if expected_bsz == 0:
            expected_dim = int(distributed_utils.all_reduce(
                reshaped_input_shape[0] * torch.ones((1,), dtype=torch.long, device=input.device),
                group=dist.group.WORLD,
                op="max",
            ).item())
            padded_input = torch.zeros(
                (expected_dim, reshaped_input_shape[1]),
                dtype=input.dtype, layout=input.layout, device=input.device)
            padded_input[:reshaped_input_shape[0], :] = reshaped_input
            reshaped_input = padded_input

            padded_input_padding_mask = torch.ones(
                (expected_dim,), dtype=torch.bool, device=padded_input.device
            )
            if reshaped_input_padding_mask is not None:
                padded_input_padding_mask[:reshaped_input_shape[0]] = reshaped_input_padding_mask
            else:
                padded_input_padding_mask[:reshaped_input_shape[0]] = False
            reshaped_input_padding_mask = padded_input_padding_mask

        if has_tutel:
            l_aux, self.metadata, C, E, indices_, locations_, gates_ = self.gate(reshaped_input, reshaped_input_padding_mask)
            S, M = reshaped_input.size(0), reshaped_input.size(1)

            if not hasattr(self, '_tutel_dispatcher'):
                self._tutel_dispatcher = tutel_moe.fast_dispatcher(E, C, M, dispatch_dtype=reshaped_input.dtype)
            self._tutel_dispatcher.update(indices_, locations_, gates_, capacity=C)
            dispatched_input = self._tutel_dispatcher.encode(reshaped_input)
        else:
            gate_type = self.args.gate_type
            # torch.cuda.synchronize()
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            l_aux, combine_weights, dispatch_mask, self.metadata, gpu_expert_assign, gather_list = \
                                            self.gate(reshaped_input, reshaped_input_padding_mask, \
                                                gate_type = gate_type, all2allgroup = self.all2all_group, \
                                                    experts=self.experts, moe_cpu = self.args.moe_cpu)
            cuda_end.record()
            # torch.cuda.synchronize()
            # self.gpu_gate_meter.update(cuda_start.elapsed_time(cuda_end))
            
            main_stream = torch.cuda.current_stream()

            if gate_type == 'ours':
                dispatched_input = torch.mm(dispatch_mask.to(input.dtype).T, reshaped_input)
                
                # if self.args.moe_cpu:
                #     first_comm_stream = self.experts._streams['communication']
                #     first_comm_stream.wait_stream(main_stream)
                #     #with torch.cuda.stream(first_comm_stream):
                #     cpu_start = time.time() * 1000
                #     with nvtx.annotate('wait prev optim'):
                #         ## Implement with the shared semaphore
                #         self.experts._wait_for_previous_optim_step()
                #     cpu_end = time.time() * 1000
                    
                #     first_exp_id = gpu_expert_assign[self.rank][0]
                #     self.experts._upload_params(self.experts.expert_list[first_exp_id].parameters())
            else:
                dispatch_mask = dispatch_mask.to(input.dtype).permute(1, 2, 0)  # S,E,C b = -> E,C,S
                E, C, S = dispatch_mask.size()
                # print(f'dispatch_mask: {dispatch_mask}, {dispatch_mask.shape}')
                M = reshaped_input.size(1)
                assert reshaped_input.size() == (S, M)
                # einsum("sec,sm->ecm")
                dispatched_input = torch.mm(dispatch_mask.view(E*C, S), reshaped_input)  # -> (E*C),M

        # print(f'reshape_input: {reshaped_input}, {reshaped_input.shape}')
        # print(f'dispatch_mask: {dispatch_mask.view(E*C, S)}, {dispatch_mask.view(E*C, S).shape}')
        # print(f'dispatched_input: {dispatched_input}, {dispatched_input.shape}')
        # print(f'my_dispatched_input: {my_matrix.to(reshaped_input.dtype).T.mm(reshaped_input)}, \
        #                                         {my_matrix.to(reshaped_input.dtype).mm(reshaped_input).shape}')
        # print(f'combine_weight: {combine_weights.view(S, E*C)}, {combine_weights.view(S, E*C).shape}')
        # assert False
        # expert_assign_shirnk_list = []
        # gpu_expert_assign_flatten = [item for sublist in gpu_expert_assign for item in sublist]
        # for exp_id in gpu_expert_assign_flatten:
        #     for idx in range(exp_id * C, (exp_id * C + gather_list[self.rank][exp_id].item())):
        #         expert_assign_shirnk_list.append(idx)
        
        # indices = torch.tensor(expert_assign_shirnk_list, device='cuda')
        # print(f'indices: {indices}')
        # dispatched_input = dispatched_input.index_select(0, index=indices)
        # print(f'dispatched_input: {dispatched_input}, {dispatched_input.shape}')

        # input_split_tensors = []
        # for output in gather_list:
        #     _ineer_sum = []
        #     for dev_id in range(len(gpu_expert_assign)):
        #         _ineer_sum.append(output[gpu_expert_assign[dev_id]].sum())
        #     input_split_tensors.append(torch.stack(_ineer_sum, dim = 0))
        
        # input_splits = torch.stack(input_split_tensors, dim = 0)
        # output_splits = input_splits.transpose(0, 1)  
        # print(input_splits, output_splits, expert_assign_shirnk_list)
        
        # print(f'combine weights: {combine_weights.view(S, E*C)}, {combine_weights.view(S, E*C).shape}')
        # combine_weights = combine_weights.view(S, E*C).index_select(1, index=indices)
        # print(f'combine weights: {combine_weights}, {combine_weights.shape}')
        
        # assert False
        # gpu_expert_assign = [[0,1], [2,3]]
        # print(gpu_expert_assign)
        
        if gate_type == 'ours':
            # torch.cuda.synchronize()
            # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            
            input_split_tensors = []
            for output in gather_list:
                _ineer_sum = []
                for dev_id in range(len(gpu_expert_assign)):
                    _ineer_sum.append(output[gpu_expert_assign[dev_id]].sum())
                input_split_tensors.append(torch.stack(_ineer_sum, dim = 0))
            
            input_splits = torch.stack(input_split_tensors, dim = 0)
            output_splits = input_splits.transpose(0, 1)

            cuda_end.record()
            # torch.cuda.synchronize()
            # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
            # self.gpu_postprocess_meter.update(cuda_start.elapsed_time(cuda_end))

            input_split, output_split = input_splits[self.rank].tolist(), output_splits[self.rank].tolist()
            
            # def input_postprocess_wrapper(_gpu_expert_assign, _gather_list, _dispatched_input, _combine_weights):
            #     # torch.cuda.synchronize()
            #     # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
            #     cuda_start = torch.cuda.Event(enable_timing=True)
            #     cuda_end = torch.cuda.Event(enable_timing=True)
            #     cuda_start.record()
                
            #     # if self.args.moe_cpu:
            #     #     first_comm_stream = self.experts._streams['communication']
            #     #     first_comm_stream.wait_stream(main_stream)
            #     #     #with torch.cuda.stream(first_comm_stream):
            #     #     cpu_start = time.time() * 1000
            #     #     with nvtx.annotate('wait prev optim'):
            #     #         ## Implement with the shared semaphore
            #     #         self.experts._wait_for_previous_optim_step()
            #     #     cpu_end = time.time() * 1000
                    
            #     #     first_exp_id = _gpu_expert_assign[self.rank][0]
            #     #     #print(f'first_exp_id: {first_exp_id}')
            #     #     self.experts._upload_params(self.experts.expert_list[first_exp_id].parameters())
                
            #     ## Rearrange dispatched_input to the assigned experts
            #     # print(gpu_expert_assign, gather_list)
            #     # gpu_expert_assign_flatten = [item for sublist in _gpu_expert_assign for item in sublist]
            #     # ## gather_list[current_rank] => [3, 2, 0, 1] // gpu_expert_assign => [[3, 2], [0, 1]]
            #     # # print(gpu_expert_assign_flatten)
                
            #     # expert_assign_shirnk_list = []

            #     # for exp_id in gpu_expert_assign_flatten:
            #     #     for idx in range(exp_id * C, (exp_id * C + _gather_list[self.rank][exp_id].item())):
            #     #         expert_assign_shirnk_list.append(idx)
            #     # print(expert_assign_shirnk_list)
            #     # # expert_assign_shirnk_list = [12, 13, 14, 0, 1, 2, 4, 5]
            #     # indices = torch.tensor(expert_assign_shirnk_list, device='cuda')
            #     # dispatched_input_temp = _dispatched_input.index_select(0, index=indices)
            #     # del _dispatched_input
            #     # combine_weights_temp = _combine_weights.view(S, E*C).index_select(1, index=indices)
            #     # del _combine_weights
            #     # expert_assign_shirnk_mat = one_hot(torch.tensor(expert_assign_shirnk_list).to('cuda'), \
            #     #                                             num_classes=E*C, unsqueeze_indices=True).to(torch.float16)
            #     # # print(f'expert_assing_shirnk_mat: {expert_assign_shirnk_mat}, {expert_assign_shirnk_mat.shape}')
            #     # dispatched_input = torch.mm(expert_assign_shirnk_mat, _dispatched_input)
            #     # # print(f'dispatched_iput: {dispatched_input}, {dispatched_input.shape}')
            #     # # assert dispatched_input.shape == reshaped_input.shape
                
            #     # ## Rearrange combine_weights for the future combined_output.
            #     # # print(f'combine_weight: {combine_weights}, {combine_weights.shape}')
            #     # combine_weights_temp = _combine_weights.permute(1, 2, 0)
            #     # combine_weights_temp = torch.mm(expert_assign_shirnk_mat, combine_weights_temp.view(E*C, S))
            #     # print(f'{self.rank} {combine_weights_temp.dtype} combine_weights_temp: {torch.norm(combine_weights_temp, p=2)}')
                
            #     ## Making splits for All to All
                
            #     # print(_gpu_expert_assign, _gather_list, _gather_list[0][_gpu_expert_assign[0]])
                
            #     # input_splits = []
            #     # for output in _gather_list:
            #     #     split = []
            #     #     for expert_list in _gpu_expert_assign:
            #     #         val = 0
            #     #         for exp_id in expert_list:
            #     #             val +=  output[exp_id].item()
            #     #         split.append(val)
            #     #     input_splits.append(split)
                
            #     # output_splits = torch.tensor(input_splits).to('cuda')
                
            #     # # output_splits = copy.deepcopy(input_splits)
            #     # # output_splits = list(map(list, zip(*output_splits)))
            #     # print(input_splits, output_splits)
                
            #     input_split_tensors = []
            #     for output in _gather_list:
            #         _ineer_sum = []
            #         for dev_id in range(len(_gpu_expert_assign)):
            #             _ineer_sum.append(output[_gpu_expert_assign[dev_id]].sum())
            #         input_split_tensors.append(torch.stack(_ineer_sum, dim = 0))
                
            #     input_splits = torch.stack(input_split_tensors, dim = 0)
            #     output_splits = input_splits.transpose(0, 1)  

            #     # print(input_splits, output_splits, self.rank)
            #     # input_split, output_split = input_splits[self.rank].tolist(), output_splits[self.rank].tolist()
                
            #     cuda_end.record()
            #     # torch.cuda.synchronize()
            #     # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
            #     self.gpu_postprocess_meter.update(cuda_start.elapsed_time(cuda_end))

            #     return input_splits, output_splits
            
            # input_splits, output_splits = input_postprocess_wrapper(gpu_expert_assign, gather_list, dispatched_input, combine_weights)
            # input_split, output_split = input_splits[self.rank].tolist(), output_splits[self.rank].tolist()
        else:
            input_split, output_split = None, None
        
        # print(input_split, output_split)
        # print(f'post_dispatched_input: {dispatched_input}, {dispatched_input.shape}')
            
        if self.all2all_size > 1:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            # torch.cuda.synchronize()
            cuda_start.record()
            dispatched_input = self.all_to_all_wrapper(dispatched_input, input_split, output_split)
            cuda_end.record()
            # torch.cuda.synchronize()
            # self.gpu_alltoall_meter.update(cuda_start.elapsed_time(cuda_end))
            
            # def alltoallwrapper(all_to_all, _input_splits = None, _output_splits = None):
            #     # torch.cuda.synchronize()
            #     # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
            #     cuda_start = torch.cuda.Event(enable_timing=True)
            #     cuda_end = torch.cuda.Event(enable_timing=True)
            #     cuda_start.record()
                
            #     all_to_all = self.all_to_all_wrapper(all_to_all, _input_splits, _output_splits)
                
            #     cuda_end.record()
            #     # torch.cuda.synchronize()
            #     # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
            #     # self.cpu_alltoall_meter.update((cpu_end - cpu_start))
            #     self.gpu_alltoall_meter.update(cuda_start.elapsed_time(cuda_end))
            #     return all_to_all
            
            # #print(f'dispatched_input: {dispatched_input.to(torch.float64).sum()}, {dispatched_input.shape}')
            # with nvtx.annotate('pre forward alltoall'):
            #     dispatched_input = alltoallwrapper(dispatched_input, input_split, output_split)
        # rank = distributed_utils.get_global_rank()
        # print(f'expert_input: {dispatched_input}, {dispatched_input.shape}')
        
        if gate_type == 'ours' or gate_type =='baseline':
            # print(gather_list, input_split, output_split)
            
            drop_computation = reshaped_input.shape[0]
            
            if gate_type == 'ours':
                baseline_computation = torch.max(sum(gather_list)).item() * self.num_local_experts
                baseline_nonzero_computation = sum(sum(gather_list)[self.rank * self.num_local_experts: (self.rank + 1) * self.num_local_experts])
                ours_computation = torch.max(output_splits.sum(dim=1))
                self.ours_meter.update(ours_computation.item())
                self.baseline_nonzero_meter.update(baseline_nonzero_computation.item())
            elif gate_type == 'baseline':
                baseline_computation = dispatched_input.shape[0]
                
            self.token_drop_meter.update(drop_computation)
            self.baseline_meter.update(baseline_computation)
            
            
            # print(output_splits, output_splits.sum(dim=1), torch.max(output_splits.sum(dim=1)), torch.min(output_splits.sum(dim=1)))
            # print(gather_list, sum(gather_list))
            # print(f'{self.rank}, {baseline_computation}')
            # print(input_split, output_split)
            # print((baseline_computation - ours_computation) / baseline_computation, baseline_computation, ours_computation)
            # self.improvement_meter.update((baseline_computation - ours_computation) / baseline_computation)
            # self.degradation_meter.update((ours_computation - S) / ours_computation)
            
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        
        # torch.cuda.synchronize()
        # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
        cuda_start.record()
        
        ## Preparing Chunks
        if gate_type == 'ours':
            if self.args.moe_cpu:
                first_exp_id = gpu_expert_assign[self.rank][0]
                first_comp_stream = self.experts.comp_streams[first_exp_id]
            else:
                first_comp_stream = torch.cuda.current_stream()
                
            first_comp_stream.wait_stream(main_stream)
            with torch.cuda.stream(first_comp_stream):
                ## split need to be changed according to the num_local_experts
                ## Originally, the dispatched_input reshaped into 2 steps, all2all-size -> num_local experts
                dispatched_input_all2all = dispatched_input.split(output_split, dim = 0)

                split_chunks = []
                for idx, each_input in enumerate(dispatched_input_all2all):
                    split_list = [gather_list[idx][exp_id] for exp_id in gpu_expert_assign[self.rank]]
                    split_chunks.append(list(each_input.split(split_list, dim = 0)))
                
                chunks = []
                for i in range(self.num_local_experts):
                    chunks.append(torch.cat([row[i] for row in split_chunks], dim = 0))
        else:
            dispatched_input = dispatched_input.reshape(self.all2all_size, self.num_local_experts, -1, d_model)
            chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        
        ## Expert Forward
        if self.args.moe_cpu:
            expert_outputs, last_comp_stream = self.experts(chunks, gpu_expert_assign, self.rank)
        else:
            expert_outputs = []
            for chunk, expert in zip(chunks, self.experts):
                ## NOTE
                ## The bias term adds some value on zero padding
                ## So, exact value would not match.
                expert_outputs += [expert(chunk)]
            last_comp_stream = torch.cuda.current_stream()

        if gate_type == 'ours':
            with torch.cuda.stream(last_comp_stream):
                expert_output = torch.cat(expert_outputs, dim=0)
                def _backward_stream_connection(*unsued):
                    last_comp_stream.wait_stream(main_stream)
                if expert_output.requires_grad:
                    expert_output.register_hook(_backward_stream_connection)
            main_stream.wait_stream(last_comp_stream)
            expert_output_split = []
            for exp_id in gpu_expert_assign[self.rank]:
                for gate_output in gather_list:
                    expert_output_split.append(gate_output[exp_id].item())
            
            expert_output_list = []
            expert_output = expert_output.split(expert_output_split, dim = 0)
            for i in range(self.all2all_size):
                #print([expert_output[j].shape for j in range(i, len(expert_output_split), self.all2all_size)])
                expert_output_list.append(torch.cat([expert_output[j] for j in \
                                range(i, len(expert_output_split), self.all2all_size)], dim = 0))
            expert_output = torch.cat(expert_output_list, dim = 0)
        else:
            expert_output = torch.cat(expert_outputs, dim=1)
        # print(f'expert_ouput: {expert_output}, {expert_output.shape}')
        cuda_end.record()
        # torch.cuda.synchronize()
        # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
        cpu_end = time.time() * 1000
        # self.cpu_expert_meter.update((cpu_end - cpu_start))
        # self.gpu_expert_meter.update(cuda_start.elapsed_time(cuda_end))

        # def expertwrapper(dispatched_input):
        #     cuda_start = torch.cuda.Event(enable_timing=True)
        #     cuda_end = torch.cuda.Event(enable_timing=True)
            
        #     # torch.cuda.synchronize()
        #     # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
        #     cuda_start.record()
        #     # Re-shape after all-to-all: ecm -> gecm
            
        #     ## Need to reshape the output to expert specific size
        #     if gate_type == 'ours':
                
        #         if self.args.moe_cpu:
        #             first_exp_id = gpu_expert_assign[self.rank][0]
        #             first_comp_stream = self.experts.comp_streams[first_exp_id]
        #         else:
        #             first_comp_stream = torch.cuda.current_stream()
                    
        #         first_comp_stream.wait_stream(main_stream)
        #         with torch.cuda.stream(first_comp_stream):
        #             ## split need to be changed according to the num_local_experts
        #             ## Originally, the dispatched_input reshaped into 2 steps, all2all-size -> num_local experts
        #             dispatched_input_all2all = dispatched_input.split(output_split, dim = 0)

        #             split_chunks = []
        #             for idx, each_input in enumerate(dispatched_input_all2all):
        #                 split_list = [gather_list[idx][exp_id] for exp_id in gpu_expert_assign[self.rank]]
        #                 split_chunks.append(list(each_input.split(split_list, dim = 0)))
                    
        #             chunks = []
        #             for i in range(self.num_local_experts):
        #                 chunks.append(torch.cat([row[i] for row in split_chunks], dim = 0))
                    
        #         # for chunk in chunks:
        #         #     print(chunk.shape, self.rank)
        #         # print(f'chunk: {chunks[0].shape}')
        #     else:
        #         dispatched_input = dispatched_input.reshape(self.all2all_size, self.num_local_experts, -1, d_model)
        #         # print(f'dispatched_input: {dispatched_input.shape}')
        #         chunks = dispatched_input.chunk(self.num_local_experts, dim=1)
        #         # print(f'chunk: {chunks[0].shape}')
                
        #     # print(f'{rank} {chunks[0].dtype} chunk:', [torch.norm(c, p=2) for c in chunks])
        #     if self.args.moe_cpu:
                
        #         # def _stream_connection(*unused: Any):
        #         #     self.comp_connection_stream.wait_stream(self.experts.comp_streams[gpu_expert_assign[self.rank][0]])
        #         #     torch.cuda.current_stream().wait_stream(self.comp_connection_stream)
                    
        #         #     # print('post backward stream connection')
        #         #     # print(unused)
        #         #     # assert False
                
        #         # chunks[0].register_hook(_stream_connection)
                
        #         expert_outputs, last_comp_stream = self.experts(chunks, gpu_expert_assign, self.rank)
        #         # expert_outputs = self.experts(chunks, [[0, 1], [2, 3]], self.rank)
        #     else:
        #         expert_outputs = []
        #         for chunk, expert in zip(chunks, self.experts):
        #             # print(f'chunk: {chunk.to(torch.float64).sum()}, {chunk.shape}, {chunk.dtype}')
        #             # for param in expert.parameters():
        #             #     print(f'expert: {param.to(torch.float64).sum()}')
        #             ## NOTE
        #             ## The bias term adds some value on zero padding
        #             ## So, exact value would not match.
        #             # if gate_type == 'ours':
        #             #     chunk = torch.cat([chunk, torch.zeros(56, 64, device = chunk.device, dtype=chunk.dtype)], dim = 0)
        #             # with torch.no_grad():
        #             #     print(f'exp:', [param.norm() for param in list(expert.parameters())])
        #             # print(f'{self.rank}  {list(expert.parameters())[0].dtype} !!!exp params: ', [torch.norm(ep, p=2, dtype=torch.float32) for ep in expert.parameters()])
        #             # print(f'{self.rank}, shape: {chunk.shape}')
        #             expert_outputs += [expert(chunk)]
        #             # print(f'{self.rank} {expert_outputs[-1].dtype} output: {torch.norm(expert_outputs[-1], p=2, dtype=torch.float32)}')
        #             # print(f'each_expert_output: {expert_outputs[-1].to(torch.float64).sum()}, {expert_outputs[-1].shape}')
        #         last_comp_stream = torch.cuda.current_stream()
        #     # print(expert_outputs)

        #     # print(f'{self.rank} expert_outputs: {expert_outputs}, {len(expert_outputs)}, {expert_outputs[0].shape}')

        #     if gate_type == 'ours':
        #         # for ep_op in expert_outputs:
        #         #     print(ep_op.shape, self.rank)
        #         # if self.args.moe_cpu:
        #         #     self.comp_connection_stream.wait_stream(comp_stream)
                
        #         with torch.cuda.stream(last_comp_stream):
        #             expert_output = torch.cat(expert_outputs, dim=0)
        #             def _backward_stream_connection(*unsued):
        #                 last_comp_stream.wait_stream(main_stream)
        #             expert_output.register_hook(_backward_stream_connection)
                    
        #         main_stream.wait_stream(last_comp_stream)
                
        #         # if self.args.moe_cpu:
                    
        #         #     ## Main Stream should wait for the self.comp_connection stream in backward
        #         #     main_stream = torch.cuda.current_stream()
        #         #     main_stream.wait_stream(self.comp_connection_stream)
                    
        #         #     def _backward_stream_connection(*unused):
        #         #         curr_stream = torch.cuda.current_stream()
        #         #         curr_stream.wait_stream(main_stream)
        #         #         # print("pre backward stream connection")
                    
        #         #     expert_output.register_hook(_backward_stream_connection)
                
        #         # print(f'{self.rank} expert_output: {expert_output}, {expert_output.shape}')
        #         # print(f'expert_ouput: {expert_output.to(torch.float64).sum()}, {expert_output.shape}')
        #         expert_output_split = []
        #         for exp_id in gpu_expert_assign[self.rank]:
        #             for gate_output in gather_list:
        #                 expert_output_split.append(gate_output[exp_id].item())
                
        #         #print(expert_output_split)
        #         expert_output_list = []
        #         expert_output = expert_output.split(expert_output_split, dim = 0)
        #         for i in range(self.all2all_size):
        #             #print([expert_output[j].shape for j in range(i, len(expert_output_split), self.all2all_size)])
        #             expert_output_list.append(torch.cat([expert_output[j] for j in \
        #                             range(i, len(expert_output_split), self.all2all_size)], dim = 0))
        #         expert_output = torch.cat(expert_output_list, dim = 0)
        #     else:
        #         expert_output = torch.cat(expert_outputs, dim=1)
        #     # print(f'expert_ouput: {expert_output.to(torch.float64).sum()}, {expert_output.shape}')
        #     cuda_end.record()
        #     # torch.cuda.synchronize()
        #     # print(torch.cuda.memory_reserved() / 1024 / 1024 / 1024)
            
        #     cpu_end = time.time() * 1000
        #     # self.cpu_expert_meter.update((cpu_end - cpu_start))
        #     self.gpu_expert_meter.update(cuda_start.elapsed_time(cuda_end))
        #     return expert_output

        # expert_output = expertwrapper(dispatched_input)
        # print(f'expert_output: {expert_output} {expert_output.shape}')
        #assert False
        
        if self.all2all_size > 1:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            # torch.cuda.synchronize()
            cuda_start.record()
            expert_output = self.all_to_all_wrapper(expert_output, output_split, input_split)
            cuda_end.record()
            # torch.cuda.synchronize()
            # self.gpu_alltoall_meter.update(cuda_start.elapsed_time(cuda_end))
            
            # print(f'expert_output: {expert_output.shape}')
            #assert False
            # print(output_split, input_split)
            # with nvtx.annotate('post forward all2all'):
            #     expert_output = alltoallwrapper(expert_output, output_split, input_split)
            #expert_output = self.all_to_all_wrapper(expert_output)

        # print(f'{rank} expert_ouput after all to all: {torch.norm(expert_output, p=2)}')
        # print(f'post_expert_output: {expert_output} {expert_output.shape}')
        if gate_type == 'ours':
            # print(f'combined_weights: {combine_weights} {combine_weights.shape}')
            combined_output = combine_weights.mm(expert_output)
        else:
            # Re-shape back: gecm -> ecm
            expert_output = expert_output.reshape(self.all2all_size * self.num_local_experts, -1, d_model)
            
            # print(f'expert_ouput: {expert_output.shape}')
            # print(f'combine_weights: {combine_weights.view(S, E*C).shape}')
            if has_tutel:
                combined_output = self._tutel_dispatcher.decode(expert_output.view(E*C, M))
            else:
                # einsum("sec,ecm->sm")
                # print(f'combined_weights: {combine_weights.to(torch.float64).sum()}, {combine_weights.shape}')
                combined_output = combine_weights.view(S, E*C).mm(expert_output.view(E*C, M))

        # print(f'{rank} combined_output: {torch.norm(combined_output, p=2)}')
        # print(f'combined_output: {combined_output}, {combined_output.shape}')
        # assert False
        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[:reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input_shape[0], :, :]

        # print(f'{rank} final combined_output: {torch.norm(combined_output, p=2)}')
        self.record_all_to_all_stats()
        # assert False
        
        def _backward_pre_expert_upload(*unused):
            last_exp_id = gpu_expert_assign[self.rank][-1]
            self.experts._streams['communication'].wait_stream(main_stream)
            self.experts._upload_params(self.experts.expert_list[last_exp_id].parameters())
        
        if self.args.moe_cpu and self.args.checkpoint_activations and combined_output.requires_grad:
            combined_output.register_hook(_backward_pre_expert_upload)
        # print(f'mem_info: {torch.cuda.mem_get_info()}')
    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor,  input_splits, output_splits):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        output = _AllToAll.apply(self.all2all_group, input, input_splits, output_splits)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += (cpu_end - cpu_start)
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self):
        # controlled via an argument as we want to minimize any impact from # # # torch.cuda.synchronize()
        record_a2a_perf_stats = getattr(self.args, 'record_a2a_perf_stats', False)
        if record_a2a_perf_stats:
            # # # torch.cuda.synchronize()
            self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
        # reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []
