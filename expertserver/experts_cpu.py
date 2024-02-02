import torch
import functools
from torch.nn.parameter import Parameter
from torch.cuda.streams import Stream
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)
import expertserver.storage_memory_utils
from shared_pinned_memory import shared_pinned_memory
import expertserver.utils
from expertserver.utils import ParamType, Optim_TaskType
from fairseq import utils as fairseq_utils
import nvtx

class Experts(torch.nn.Module):
    def __init__(self, args, rank, expert_layer_cls, layer):
        super().__init__()
        self.rank = rank
        self.mixed_precision = args.fp16
        self.compute_dtype = torch.float16 if self.mixed_precision else torch.float32
        
        embed_dim = args.decoder_embed_dim or args.encoder_embed_dim 
        ffn_dim = args.decoder_ffn_embed_dim or args.encoder_ffn_embed_dim
        self.expert_list = []
        
        # storage_memory_utils.see_memory_usage(0)
        self.num_experts = args.moe_expert_count
        self.num_local_experts = args.moe_expert_count // args.world_size
        for i in range(self.num_experts):
            with fairseq_utils.set_torch_seed(960502 + self.rank * self.num_experts + i):
                expert = expert_layer_cls(args, embed_dim=embed_dim, ffn_dim = ffn_dim)
            self.expert_list.append(expert)
        
        # storage_memory_utils.see_memory_usage(0)
        self.args = args
        self.layer = layer
        self.move_params_to_cpu = self.move_grads_to_cpu = args.moe_cpu
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self.update_freq = expertserver.utils.global_cfg.optimization.update_freq[0] if expertserver.utils.global_cfg else 1
        # print(f'self.update_freq: {self.update_freq}')
        self.curr_freq = 0
        
    def set_optimizer(self, optim) -> None:
        self.optimizer = optim
            
    def _setup_streams(self) -> None:
        if len(self._streams) > 0:
            return
        
        if torch.cuda.is_available():
            self._streams["communication"] = torch.cuda.Stream()
            ##self._streams["optimization"] = torch.cuda.Stream()
            ## TODO: Change to self.num_local_experts
            # self.optim_streams = [Stream() for _ in range(self.num_experts)]
            self.comp_streams = [Stream() for _ in range(self.num_experts)]
            self.post_backward_streams = [Stream() for _ in range(self.num_experts)]
            
    @torch.no_grad()
    def _lazy_init_param_attributes(self, p: Parameter, rank: int, layer:int, expert:int, order:int):
        if hasattr(p, "_fp32"):
            return
        
        p._fp32 = p.data

        if self.mixed_precision:
            assert p._fp32.dtype == torch.float32, self

        if self.move_params_to_cpu:
            assert p._fp32.device == torch.device("cpu"), self

            # If we plan to keep the FP32 parameters on CPU, then pinning
            # memory allows us to later use non-blocking transfers when moving
            # the FP32 param shard to compute_device.
            #p._fp32 = p._fp32.pin_memory()
            p._fp32 = shared_pinned_memory(p._fp32, rank, layer, expert, order, ParamType.PARAM, True)
            p.data = p._fp32

        if self.move_params_to_cpu or self.mixed_precision:

            # In mixed precision mode, we maintain a reduced precision
            # (typically FP16) parameter shard on compute_device for performing
            # the computation in the forward/backward pass. We resize the
            # storage to size 0 at init (here) and re-materialize (by copying
            # from _fp32_shard) as needed. If offloading params to CPU, the
            # dtype of the fp16 shard will depend on the *`compute_dtype`*.
            p._fp16 = torch.zeros_like(p._fp32, device=self.compute_device, dtype=self.compute_dtype)
            expertserver.storage_memory_utils.free_storage_(p._fp16)
        
        if self.move_grads_to_cpu:
            # We can optionally move the grad shard to CPU during the backward
            # pass. In this case, it's important to pre-allocate the CPU grad
            # shard in pinned memory so that we can do a non-blocking transfer.
            # This is only needed during training and not evaluation.
            #p._cpu_grad = torch.zeros_like(p.data, device="cpu").pin_memory()
            p._cpu_grad = shared_pinned_memory(p.data, rank, layer, expert, order, ParamType.GRAD, True)
        
        # ## TODO
        # ## it should be update_freq > 1.
        # if self.update_freq > 1:
        #     p._cpu_accum_grad = torch.zeros_like(p.data, device="cpu")
        # else:
        #     p._cpu_accum_grad = p._cpu_grad
            
    def lazy_init(self) -> None:
        """Initialization steps that should happen lazily, typically right
        before the first forward pass.
        """
        # Initialize param attributes lazily, in case the param's dtype or
        # device changes after __init__.
        
        ## All GPU needs this
        ## Compute Device would be different after spawn
        self.compute_device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
        
        ## Need to initialize all experts since each GPU would eventually have all experts on their device
        for exp_id, expert in enumerate(self.expert_list):
            for param_id, p in enumerate(expert.parameters()):
                self._lazy_init_param_attributes(p, self.rank, self.layer, exp_id, param_id)
        
        ## ALL GPU needs this
        self._setup_streams()
        
        ## To get the callback
        # torch._C._activate_cuda_trace()
    
    def _wait_for_previous_optim_step(self) -> None:
        """
        The outer-most :class:`FullyShardedDataParallel` instance (i.e., the root
        instance) needs to synchronize with the default stream to ensure the
        previous optimizer step is done.
        """
        if not torch.cuda.is_available():
            return

        # cuda_start = torch.cuda.Event(enable_timing=True)
        # cuda_end = torch.cuda.Event(enable_timing=True)
        # cuda_start.record()

        if self.optimizer.scaler is None:
            self.optimizer.scaler = expertserver.utils.global_scaler
        
        for exp_id, thread in enumerate(self.optimizer.thread_list):
            if thread:
                # print(f'{self.rank}, {i} thread joining')
                thread.join()
                self.optimizer.optim_expert_events[exp_id].wait()
                self.optimizer.set_grad_to_none(exp_id)
            # elif self.curr_freq == self.update_freq and not self.optimizer.overflow:
            #     self.optimizer.increment_step(i)
        
        # print('join finished')
        
        # with nvtx.annotate('empty_cache'):
        #     torch.cuda.empty_cache()
        
        self.optimizer.thread_list = [None for _ in range(self.num_experts)]
        
        if self.curr_freq == 1:
            # print(f"expert new lr: {expertserver.utils.global_lr}")
            self.optimizer.set_lr(expertserver.utils.global_lr)
        elif self.curr_freq == self.update_freq:
            self.optimizer._multiply_factor = 1.0 / float(self.optimizer.scaler.loss_scale)
            self.optimizer.overflow = False
            self.optimizer.mul_fac_modified = False

        # print('join finished')
        
    @torch.no_grad()
    def _upload_params(self, params: List[Parameter]) -> None:
        with torch.cuda.stream(self._streams['communication']):
            for p in params:
                assert p._fp16 is not None
                ## TODO ##
                ## Need to find out why the alloc size should be the fp32
                expertserver.storage_memory_utils.alloc_storage_(p._fp16, size=p._fp32.size())
                p._fp16.copy_(
                    # If move_params_to_cpu is True, this will be non-blocking
                    # because _fp32 is pinned, otherwise it's a no-op.
                    p._fp32.to(p._fp16.device, non_blocking=True)
                )
                # with torch.no_grad():
                #     print(p._fp16.norm())
                p.data = p._fp16
                # with torch.no_grad():
                #     print(p.data.norm())

    @torch.no_grad()
    def _free_fp16_param(self, params: Optional[List[Parameter]] = None) -> None:
        """Free storage for FP16 for a list of params."""
        current_stream = torch.cuda.current_stream()
        for p in params:
            if p._fp16 is not None:
                # _fp16 is allocated in "fp32_to_fp16" stream, so we can't
                # free it until the work in the current stream completes.
                
                ## Ensures that the tensor memory is not reused for another 
                ## tensor until all current work queued on stream are complete.
                p._fp16.record_stream(current_stream)
                expertserver.storage_memory_utils.free_storage_(p._fp16)

    @torch.no_grad()
    def _use_fp32_param(self, params: Optional[List[Parameter]] = None) -> None:
        """Use FP32 for a list of params."""
        for p in params:
            p.data = p._fp32
            
    def _register_pre_backward_hooks(self, outputs: Any, idx, exp_id_list) -> Any:
        """Register pre-backward hook to run before the wrapped module's
        backward. Hooks should be attached to all outputs from the forward.
        Returns:
            outputs: new outputs with hooks registered if they requires gradient.
        """
        if not torch.is_grad_enabled():
            return # don't register hooks if grad isn't enabled

        # self._post_backward_callback_queued = False

        def _pre_backward_hook(*unused: Any) -> None:
            
            # print(list(self.expert_list[prev_exp_id].parameters())[0].dtype, list(self.expert_list[prev_exp_id].parameters())[0].device)
            # print(f'prev: {list(self.expert_list[prev_exp_id].parameters())[0].dtype}, {list(self.expert_list[prev_exp_id].parameters())[0].device}')
            
            curr_stream = torch.cuda.current_stream()
           
            curr_exp_id, prev_exp_id = exp_id_list[idx], exp_id_list[idx - 1]
           
            # print(f'--pre_backward {curr_exp_id}th expert hook fired')

            # if idx == len(exp_id_list) - 1:
            #     ## Leave this
            #     curr_stream.wait_stream(comp_connection_stream)
                
            #     ## TODO
            #     ## Move this to previous output for more overlap
            #     # self._upload_params(self.expert_list[curr_exp_id].parameters())
            #     self._streams['communication'].wait_stream(comm_connection_stream)
            # else:
            #     next_exp_id = exp_id_list[idx + 1]
            #     curr_stream.wait_stream(self.comp_streams[next_exp_id])

            if idx != len(exp_id_list) - 1:
                next_exp_id = exp_id_list[idx + 1]
                curr_stream.wait_stream(self.comp_streams[next_exp_id])
            
            ## Wait for the current expert
            curr_stream.wait_stream(self._streams['communication'])
            
            if idx != 0:
                # if idx == len(exp_id_list) - 1:
                #     self._streams['communication'].wait_stream(comp_connection_stream)
                # else:
                if idx != len(exp_id_list) - 1:
                    self._streams['communication'].wait_stream(self.comp_streams[exp_id_list[idx + 1]])
                self._upload_params(self.expert_list[prev_exp_id].parameters())
        
        # Attach hooks to Tensor outputs.
        if outputs.requires_grad:
            outputs.register_hook(_pre_backward_hook)

        # print(f'--pre_backward {curr_exp_id}th expert hook attached')

    def _register_post_backward_hooks(self, params: Optional[List[Parameter]], idx, exp_id_list) -> None:
        if not torch.is_grad_enabled():
            return  # don't register grad hooks if grad isn't enabled
        
        # print(f'{next(params).dtype}, {next(params).device}')
        # print(f'Experts_{exp_id} post hook registerd')
        # post_backward_stream = torch.cuda.Stream()
        # self.post_backward_streams.append(post_backward_stream)
        #print(f'register // exp_id: {exp_id} -> {post_backward_stream}')
        for i, p in enumerate(params):
            #print(f'Experts_{exp_id} {i}th param post hook registerd')
            if p.requires_grad:
                ## The goal is to attach a hook
                ## on each of the parameter's gradient generating function (``grad_acc``
                ## below) so that the hook is called *after* all gradients for that
                ## param are computed.
                p_tmp = p.expand_as(p)  # Get a grad_fn on p_tmp.
                assert p_tmp.grad_fn is not None
                grad_acc = p_tmp.grad_fn.next_functions[0][0]  # Gets its GradAccumulation object.
                # print(p_tmp.grad_fn, p_tmp.grad_fn.next_functions, grad_acc, grad_acc.next_functions)
                # assert False
                handle = grad_acc.register_hook(functools.partial(self._post_backward_hook, p, idx, exp_id_list, i))
                # Important, we need to save the hook, otherwise, it appears to be
                # deleted/freed/unregistered.
                # However, we don't free/unhook at the end of bwd (as we used to do it
                # in _finalize_parameters below). If we do, that may unregister the wrong hook.
                p._shard_bwd_hook = (grad_acc, handle)

    @torch.no_grad()
    def _post_backward_hook(self, param: Parameter, idx, exp_id_list, partition_id, *unused: Any) -> None:
        # cuda_start = torch.cuda.Event(enable_timing=True)
        # cuda_end = torch.cuda.Event(enable_timing=True)
        # cuda_start.record()
        # print(f'{unused[1][0].device}')
        # print(f'{param.dtype}, {param.device}')
        #
        #torch.cuda.synchronize()
        # with nvtx.annotate(f'main_stream_sync{exp_ids[0]}'):
        #     torch.cuda.synchronize()
        # main_stream.synchronize()
        
        if param.grad is None:
            return
            
        if param.grad.requires_grad:
            raise RuntimeError("Only works with gradients that don't require gradients")
        
        # with nvtx.annotate(f"param print {exp_ids}"):
        #     print(param.grad)
        # print(f'post_backward hooks fired, {exp_ids} {torch.cuda.current_stream()}')
        
        ## Gradients from 'unused' should be ready, not param.grad.data
        # print(f'{self.rank} post backward unused {unused[1][0]}')
        
        # if partition_id == 0:
            #     with nvtx.annotate('main_stream_sync'):
            #         main_stream.synchronize()
            #     with nvtx.annotate(f'clip_grad_norm{exp_ids[0]}'):
            #         grad_norm = self.optimizer.clip_grad_norm(exp_id, post_backward_stream)
            #     try:
            #         print(f'!!!grad_norm {grad_norm}, {self.optimizer._multiply_factor}')
            #         total_norm = grad_norm * self.optimizer._multiply_factor
            #         print(f'!!!total_norm {total_norm}')
            #         self.optimizer.scaler.check_expert_overflow(total_norm)
                    
            #         if not torch.isfinite(total_norm).all():
            #             # check local gradnorm single GPU case, trigger NanDetector
            #             raise FloatingPointError("gradients are Nan/Inf")
                    
            #     except FloatingPointError:
            #         raise
            #     except OverflowError:
            #         print('overflow', exp_ids)
            #         self.optimizer.overflow = True
            #         # self.optimizer.set_grad_to_none(exp_id)
        
        # print(param.grad)
        
        if self.mixed_precision:
            self._free_fp16_param([param])
        ## Switch to FP32 shard after backward for optimization.
        self._use_fp32_param([param])
        
        # print(f'param.grad.data {exp_ids}: {torch.norm(param.grad.data, p=2, dtype=torch.float32)}')

        curr_exp_id = exp_id_list[idx]
        post_backward_stream = self.post_backward_streams[curr_exp_id]
        
        if idx != len(exp_id_list) - 1:
            post_backward_stream.wait_stream(self.post_backward_streams[exp_id_list[idx + 1]])
            
        # print(exp_ids, post_backward_stream, self.comp_streams[exp_id])
        with torch.cuda.stream(post_backward_stream):
            post_backward_stream.wait_stream(self.comp_streams[curr_exp_id])
            
            if not self.optimizer.mul_fac_modified:
                c = self.args.world_size / (self.args.batch_size * self.args.tokens_per_sample * self.args.world_size)
                # print(self.args.world_size, self.args.batch_size * self.args.tokens_per_sample)
                self.optimizer.set_multiply_factor(c)
            
            ####
            # with nvtx.annotate(f'clip_grad_norm {exp_ids}'):
            #     ## Multiply grads // Set multiply grad once per iteration
            #     ## Calculating each param's grad norm
            #     # print(unused)
            #     # if self.optimizer.global_step < 100:
            #     @torch.no_grad()
            #     def _clip_grad_norm(_grad):
            #         return torch.norm(_grad, p=2, dtype=torch.float32)
            #     grad_norm = _clip_grad_norm(param.grad) * self.optimizer._multiply_factor
            #     try:
            #         #self.optimizer.scaler.check_expert_overflow(grad_norm)
            #         if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            #             raise OverflowError
            #         # if not torch.isfinite(grad_norm).all():
            #         #     # check local gradnorm single GPU case, trigger NanDetector
            #         #     raise FloatingPointError("gradients are Nan/Inf")
            #     except OverflowError:
            #         print('expert overflow', exp_ids)
            #         self.optimizer.overflow = True
            #         self.optimizer.set_grad_to_none(exp_id)
            ###
        
            # print(f'param.grad.data {exp_ids} {self.comp_streams[exp_id]}: {torch.norm(param.grad.data, p=2, dtype=torch.float32)}')
            
            # with nvtx.annotate(f'clip_grad_norm {exp_ids}'):
            #     with torch.no_grad():
            #         grad_norm = torch.norm(unused[1][0], p=2, dtype=torch.float32) * self.optimizer._multiply_factor
            #     print(f'{self.rank} grad_norm: {grad_norm}')
            #     if grad_norm == float("inf") or grad_norm != grad_norm:
            #         self.overflow = True
                    
            # Cast grad to param's dtype (typically FP32). Note: we do this
            # before the move_grads_to_cpu step so that this entire hook remains
            # non-blocking. The downside is a bit more D2H transfer in that case
           ##  Unscale the grads by multiplying the multipyling factor to grads
            # if not self.optimizer.overflow:
            with nvtx.annotate('grad comm.'):
                if self.mixed_precision:
                    # with nvtx.annotate('print'):
                    #     print(f'{self.rank} param.grad.data: {param.grad.data}')
                    orig_param_grad_data = param.grad.data
                    # print(f'param.grad.data {exp_ids}: {torch.norm(param.grad.data, p=2, dtype=torch.float32)}')
                    param.grad.data = param.grad.data.to(dtype=param.data.dtype)
                    # print(f'{self.rank} param.grad.data: {param.grad.data}')
                    # Don't let this memory get reused until after the transfer.
                    orig_param_grad_data.record_stream(post_backward_stream)
                                    
                ## Sync. Error -> Call with finalize grads
                # self.optimizer.multiply_grads(param)
                param.grad.data.mul_(self.optimizer._multiply_factor)
                
                # print(torch.norm(param.grad, p=2, dtype=torch.float32))
                # param.grad.data.mul_(self.optimizer._multiply_factor)
                # print(f'{self.rank} after mul: {torch.norm(param.grad, p=2, dtype=torch.float64)}')
                
                # Optionally move gradients to CPU, typically used if one is running the optimizer on the CPU. Once the full
                # backwards pass completes, we will set `.grad` to the CPU copy.
                if self.move_grads_to_cpu:
                    ## To do the print call, it needs to be synchornized with CPU so it is waiting for the parma.grad field
                    # print(f'{exp_ids[0]} cpu_grad', param.grad.sum())
                    # print(f'param.grad.data {exp_ids}: {torch.norm(param.grad.data, p=2, dtype=torch.float32)}')
                    param._cpu_grad.copy_(param.grad.data, non_blocking=True)
                    # print(f'{self.rank} cpu grad: {torch.norm(param._cpu_grad, p=2, dtype=torch.float64)}')
                    # Don't let this memory get reused until after the transfer.
                    param.grad.data.record_stream(post_backward_stream)
            
        #with torch.cuda.stream(self._streams["optimization"]):
        # optim_stream = self.optim_streams[curr_exp_id]
        # optim_stream.wait_stream(post_backward_stream)
        
        if partition_id == 0:
            # with nvtx.annotate('optim step'):
            # print(self.curr_freq, self.update_freq)
            if self.curr_freq != self.update_freq:
                # print('accum')
                self.optimizer.accum_grads(curr_exp_id, post_backward_stream, self.curr_freq == 0)
            elif not self.optimizer.overflow:
                # print('step')
                self.optimizer.step(curr_exp_id, post_backward_stream)
                # self.optimizer.step_queue.put((Optim_TaskType.STEP, curr_exp_id))
                # self.optimizer.optim_event.set()
            # with nvtx.annotate("optim step"):
            # print('step called', exp_ids)
            # print(f'{self.rank} {list(self.expert_list[exp_id].parameters())[0].grad.dtype} !!! {exp_id} grads: ', [torch.norm(ep.grad, p=2, dtype=torch.float32) for ep in self.expert_list[exp_id].parameters()])
            
            
            
        #     self._streams["optimization"].wait_stream(post_backward_stream)
        #     cuda_trace.register_callback_for_cuda_memory_deallocation(callback_optimizatation(exp_id))
        
                # for p in param:
                #     #if p.grad is not None:
                #     # print('changed')
                #     print(p.shape, param.shape)
                #     break
            
        # with torch.cuda.stream(self._streams["optimization"]):
        #     self._streams["optimization"].wait_stream(self._streams["post_backward"])
        #     if exp_ids[1] == 0:
        #         print(f'optimize {exp_ids[0]}')
        #         print(f'param.grad :{param.grad.device}, param._cpu_grad: {param._cpu_grad.device}, {param.grad is param._cpu_grad}')
        #         self.optimizer.step(exp_ids[0], self._streams["optimization"])
       
        ## Only do the optimization when all post_backward call for a single expert is fired.
        # with torch.cuda.stream(self._streams["optimization"]):
        #     if exp_ids[1] == 0:
        #         cuda_start = torch.cuda.Event(enable_timing=True)
        #         cuda_end = torch.cuda.Event(enable_timing=True)
        #         cuda_start.record()
        #         #print(self._streams["optimization"], torch.cuda.current_stream())
        #         self._streams["optimization"].wait_stream(self._streams["post_backward"])
        #         #self._streams["post_backward"].synchronize()
                
        #         cuda_end.record()
        #         print(cuda_start.elapsed_time(cuda_end))
        #         ## TODO
        #         ## CPU optimization
        #         # print(param.grad.device, param.device)
        #         # if optimize:
        #         #     print('optimize')
        #         #     self.optimizer.step(exp_ids[0])
        #         # else:
        #         #     print('not optimize')
        #         # if not optimize:
        #         #     self.optimizer.zero_grad(exp_ids[0])
        #         #     self.optimizer.set_grad_to_none(exp_ids[0])
                
        #         self.optimizer.step(exp_ids[0], torch.cuda.current_stream())self.optimizer.step(exp_ids[0], torch.cuda.current_stream())
                
                # self.optimizer.zero_grad(exp_ids[0])
                # self.optimizer.set_grad_to_none(exp_ids[0])
                #self._streams["optimization"].synchronize()

        # cuda_end.record()
        # print(cuda_start.elapsed_time(cuda_end))
    
    def forward(self, chunks: Union[List, Tuple], gpu_expert_assign: List, rank: int) -> torch.Tensor:
        
        ## TODO
        ## Optimize more -> Optimize wait just before uploading the params
        ## Only Stale early layers
        ## No staleness
        
        
        self.curr_freq += 1
        ## TODO
        ## Global Step should be called only once, not as many times as the number of experts
        # print(self.curr_freq, self.update_freq)
        if self.curr_freq > self.update_freq:
            self.optimizer.global_step += 1
            self.curr_freq -= self.update_freq
            
        #output = input
        # print(f'rank: {rank}, {gpu_expert_assign}')
        exp_id_list = gpu_expert_assign[rank]
        expert_outputs = []
        
        # print('-----------------expert forward---------------', exp_id_list, chunks)
        with nvtx.annotate('expert forward'):
            for idx, exp_id in enumerate(exp_id_list):
                # print(f'{self.rank} expert_id: {exp_id}')
                expert = self.expert_list[exp_id]
                ## Execute on streams['communication']
                # with torch.no_grad():
                #     print(f'--------- Before Upload {global_exp_id}---------------', [param.norm() for param in list(expert.parameters())])
                
                comp_stream = self.comp_streams[exp_id]
                
                if idx == 0:
                    pass
                elif idx == 1:
                    self._streams['communication'].wait_stream(comp_stream)
                    self._upload_params(expert.parameters())
                else:
                    self._streams['communication'].wait_stream(self.comp_streams[exp_id_list[idx - 2]])
                    self._upload_params(expert.parameters())
                
                # print(f'comp_stream{local_exp_id}: {comp_stream}')
                
                ## Execute on streams['computation']
                with torch.cuda.stream(comp_stream):
                    
                    ## Wait for the previous comp streams except for the first index
                    if idx != 0:
                        comp_stream.wait_stream(self.comp_streams[exp_id_list[idx - 1]])
                    
                    comp_stream.wait_stream(self._streams['communication'])
                    # with torch.no_grad():
                    #     print(f'--------- Upload {global_exp_id}---------------', [param.norm() for param in list(expert.parameters())])
                    self._register_post_backward_hooks(expert.parameters(), idx, exp_id_list)
                    
                    # print('local input: ', chunks[local_exp_id])
                    # for param in expert.parameters():
                    #     print(param)
                    #     break
                    # print(f'{self.layer} {exp_id} input: ', torch.norm(chunks[idx], p=2, dtype=torch.float32))
                    # print(f'{exp_id} exp params: ', [torch.norm(ep, p=2, dtype=torch.float32) for ep in expert.parameters()])
                    # print(f'{exp_id} fp 32 exp params: ', [torch.norm(ep._fp32, p=2, dtype=torch.float32) for ep in expert.parameters()])
                    # print(f'{self.rank}, shape: {chunks[idx].shape}')
                    # print(f'{idx}, {exp_id}, {exp_id_list}')
                    output = expert(chunks[idx])
                    
                    ## Removing this leads to weird gradients calculation
                    # print(f'{self.layer} {exp_id} output: ', torch.norm(output, p=2, dtype=torch.float32))
                    
                    self._free_fp16_param(expert.parameters())
                    self._use_fp32_param(expert.parameters())
                    
                    # if local_exp_id != 0:
                    self._register_pre_backward_hooks(output, idx, exp_id_list)#, \
                                            #comp_connection_stream, comm_connection_stream)
                
                    expert_outputs.append(output)
        # torch.cuda.empty_cache()
        # torch.clear_autocast_cache()
        ## The stream should be returned since the 'main' stream is different from the computation stream and
        ## the MOE_Layer forward should be aligned with computation_stream.
        return expert_outputs, comp_stream