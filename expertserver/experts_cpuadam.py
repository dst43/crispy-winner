# from deepspeed.ops.adam import DeepSpeedCPUAdam 
from torch.optim.adam import Adam
from threading import Thread
import time
import torch
import nvtx
from shared_pinned_memory import shared_pinned_memory
from expertserver.utils import ParamType, Optim_TaskType
from time import sleep
import expertserver.DeepSpeedAdam
import torch.multiprocessing as mp
import builtins
import os

@torch.no_grad()
def accumulate_grads(expert_list, exp_id, first):
    for p in expert_list[exp_id].parameters():
        if first:
            p.grad.copy_(p._cpu_grad)
        else:
            p.grad.add_(p._cpu_grad)

def set_lr(optimizers, lr):
    """Set the learning rate."""
    for optim in optimizers:
        for param_group in optim.param_groups:
            param_group["lr"] = lr
                
def optim_process_func(rank, layer, model_args, expert_cls, optim_args, optim_cls, update_freq, \
                                            optim_event, step_queue, optim_expert_events):
    if rank != 0:
        def print_pass(*args, **kwargs):
            pass
        builtins.print = print_pass
    
    ## NOTE
    ## Set the process priority based on layer.
    ## Main process has the highest priority and decreases as layers get deeper.
    
    os.sched_setaffinity(os.getpid(), [i for i in range(world_size, mp.cpu_count())])
    os.nice(max(19 - model_args.decoder_layers + layer, 19))
    
    ## Expert Initialization
    expert_list = []
    num_experts = model_args.moe_expert_count
    embed_dim = model_args.decoder_embed_dim or model_args.encoder_embed_dim
    ffn_dim = model_args.decoder_ffn_embed_dim or model_args.encoder_ffn_embed_dim
    for i in range(num_experts):
        expert = expert_cls(model_args, embed_dim=embed_dim, ffn_dim = ffn_dim)
        expert_list.append(expert)

    ## Expert params & grads into shared memory
    for exp_id, expert in enumerate(expert_list):
        for param_id, p in enumerate(expert.parameters()):
            p.data = shared_pinned_memory(p.data, rank, layer, exp_id, param_id, ParamType.PARAM, False)
            if update_freq == 1:
                p.grad = shared_pinned_memory(p.data, rank, layer, exp_id, param_id, ParamType.GRAD, False)
            else:
                p._cpu_grad = shared_pinned_memory(p.data, rank, layer, exp_id, param_id, ParamType.GRAD, False)
                p.grad = torch.zeros_like(p.data, device='cpu')
    
    ## Optimizers Initialization
    optimizers = []
    for expert in expert_list:
        optimizers.append(optim_cls(expert.parameters(), **optim_args))
    
    ## Optimizer states into shared memory
    for exp_id, optim in enumerate(optimizers):    
        for group_id, group in enumerate(optim.param_groups):
            for param_id, p in enumerate(group['params']):
                state = optim.state[p]
                
                ## TODO
                ## make the step shared tensor memory -> modify the deepsped CPU Adam, too
                state['step'] = shared_pinned_memory(torch.tensor(0.), rank, layer, exp_id, \
                                                            param_id, ParamType.OPTIM_STEP, False)

                # gradient momentum
                state['exp_avg'] = shared_pinned_memory(p.data, rank, layer, exp_id, \
                                                            param_id, ParamType.OPTIM_EXP_AVG, False)
                
                #memory_format=torch.preserve_format)
                # gradient variances
                state['exp_avg_sq'] = shared_pinned_memory(p.data, rank, layer, exp_id, \
                                                            param_id, ParamType.OPTIM_EXP_AVG_SQ, False)
    
    optim_event.set()
    optim_event.clear()

    # ## TODO
    # ## Need 3 works
    # ## 1. Experts Optim step()
    # ## 2. Experts grad accum
    # ## 3. Change the lr
    
    while True:
        optim_event.wait()
        task, add_info = step_queue.get()
        # print(f'optim process queue get: {task}, {number}')
        ## TODO
        ## not just exp_id, we need to get signal if the task is accum_grad or not
        ## or we can just do the accum throughout threading
        if task == Optim_TaskType.STEP:
            exp_id, overflow = add_info
            
            if not overflow:
                if update_freq > 1:
                    ## We need to call accum grads on step function, too.
                    accumulate_grads(expert_list, exp_id, False)
                
                optimizers[exp_id].step()
                # print(f'optim process {exp_id} step finished')
                
                ## NOTE
                ## Zero Grad would not that necessary since we are innter copying the grad values.
                # optimizers[exp_id].zero_grad()
                
        elif task == Optim_TaskType.ACCUM:
            exp_id, first = add_info
            accumulate_grads(expert_list, exp_id, first)
            
            ## Zero Grad would not that necessary since we are innter copying the grad values.
            # optimizers[exp_id].zero_grad()
            
        elif task == Optim_TaskType.LR:
            lr, _ = add_info
            exp_id = 0 
            set_lr(optimizers, lr)
        elif task == Optim_TaskType.JOIN:
            break
        else:
            assert False, f'Invalid TaskType {task}'
        
        ## Signal back to main process that this optimizer's step has been finished
        optim_expert_events[exp_id].set()
        # print(f'optim process {exp_id} event has been set')
        
        # ## Clear the event to reuse
        # optim_expert_events[exp_id].clear()
    
    # # optim_end_event.set()

class Experts_CPUAdam(object):
    def __init__(self, expert_list, opt_type, rank, layer, model_args, expert_cls, **optim_args):
        self.optimizers = []
        self.expert_list = expert_list
        self.opt_type = opt_type
        self.rank = rank
        self.layer = layer
        if self.opt_type == 'deepspeed':
            optim_args['adamw_mode'] = False
            #kwargs[]
        
        optim_cls = expertserver.DeepSpeedAdam.DeepSpeedCPUAdam if opt_type == 'deepspeed' else Adam
        self.update_freq = expertserver.utils.global_cfg.optimization.update_freq[0] if expertserver.utils.global_cfg else 1
        # for expert in expert_list:
        #     self.optimizers.append(optim_cls(expert.parameters(), **kwargs))
        
        self.optim_event = mp.Event()
        self.optim_expert_events = [mp.Event() for _ in range(len(self.expert_list))]
        self.step_queue = mp.Queue()
        self.optim_process = mp.Process(target = optim_process_func, args=(rank, layer, model_args, expert_cls, optim_args, optim_cls, self.update_freq,
                                                        self.optim_event, self.step_queue, self.optim_expert_events, ))
        self.optim_process.start()
        
        self.thread_list = []#[None for _ in range(len(expert_list))]
        self.overflow = False
        self.scaler = None
        
        self.mul_fac_modified = False
        self._multiply_factor = 1.0
        self.global_step = 0
    
    def set_grad_to_none(self, exp_id) -> None:
        ## Explicitly set the param.grad to None.
        ## If not, the param.grad would still point to _cpu_grad.
        for param in self.expert_list[exp_id].parameters():
            param.grad = None

    def _finalize_grads(self, exp_id):
        for param in self.expert_list[exp_id].parameters():    
            ## TODO
            ## CPU Comparison incurs dealy so need to find a way of comparing GPU async opration
            ## or get the grad_norm before 'post_backward' stream synchronize.
            param.grad = param._cpu_accum_grad
            
    @torch.no_grad()
    def set_multiply_factor(self, c):
        # print(f'before expert mul_fac: {self._multiply_factor}')
        self._multiply_factor *= c
        # print(f'expert mul_fac: {self._multiply_factor}')
        self.mul_fac_modified = True
    
    def terminate(self):
        self.step_queue.put((Optim_TaskType.JOIN, -1))
        self.optim_event.set()
        self.optim_process.join()
        print('optimize process terminated')
        
    ## Remove later if the process optim is working
    def set_lr(self, lr):
        """Set the learning rate."""
        # for optim in self.optimizers:
        #     for param_group in optim.param_groups:
        #         param_group["lr"] = lr
        
        self.step_queue.put((Optim_TaskType.LR, (lr, None)))
        self.optim_event.set()
        self.optim_expert_events[0].wait()
        
    ## Remove later if the process optim is working
    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizers[0].param_groups[0]["lr"]
    
    ## Remove later if the process optim is working
    def zero_grad(self, exp_id) -> None:
        ## Zero_grad is making gradients to zero. 
        ## We need this becuase currently grsad is pointing to _cpu_grad and we need to make them to be 0. 
        ## Making grad to None would not reset the _cpu_grad to 0.
        
        ## However, the cpu grad will be replaced, so we might not need this.
        self.optimizers[exp_id].zero_grad()

    ## Remove later if the process optim is working
    @torch.no_grad()
    def pre_define_optim_states(self, rank, layer):
        self.rank = rank
        self.layer = layer
        for exp_id, optim in enumerate(self.optimizers):    
            for group_id, group in enumerate(optim.param_groups):
                for param_id, p in enumerate(group['params']):
                    state = optim.state[p]
                    state['step'] = 0 if self.opt_type == 'deepspeed' else torch.tensor(0.)

                    # gradient momentum
                    state['exp_avg'] = shared_pinned_memory(p.data, rank, layer, exp_id, \
                                                                param_id, ParamType.OPTIM_EXP_AVG, False)
                    
                    # gradient variances
                    state['exp_avg_sq'] = shared_pinned_memory(p.data, rank, layer, exp_id, \
                                                                param_id, ParamType.OPTIM_EXP_AVG_SQ, False)
    @torch.no_grad()
    def increment_step(self, exp_id) -> None:
        ''' Increment other expert's step count to globally synchronize step count'''
        optim = self.optimizers[exp_id]
        for group_id, group in enumerate(optim.param_groups):
            for param_id, p in enumerate(group['params']):
                state = optim.state[p]
                state['step'] += 1
    
    @torch.no_grad()
    def clip_grad_norm(self, grad):
        return torch.norm(grad, p=2, dtype=torch.float32)
        
    @torch.no_grad()
    def _clip_grad_norm_experts(self, _exp_id):
        grad_norm = []
        for param in self.expert_list[_exp_id].parameters():
            grad_norm.append(self.clip_grad_norm(param.grad))
            
        total_norm = torch.norm(
                torch.stack(grad_norm)
            )
        return total_norm
            
    def accum_grads(self, exp_id, stream, first) -> None:
        
        def _accum_grads(_exp_id, _stream, _first):
            
            _stream.synchronize()
            
            self.step_queue.put((Optim_TaskType.ACCUM, (_exp_id, _first)))
            self.optim_event.set()
            
        thread = Thread(target=_accum_grads, args=(exp_id, stream, first))
        thread.start()
        self.thread_list[exp_id] = thread

    def step(self, exp_id, stream) -> None:
    
        # def _step(_exp_id, _stream):
            
        #     # if self.global_step < 10:
        #     #     grad_norm = self._clip_grad_norm_experts(_exp_id)

        #     with nvtx.annotate("Stream Sync"):
        #         _stream.synchronize()
            
        #     # if self.global_step < 10:
        #     #     with nvtx.annotate('overflow check'):
        #     #         if grad_norm == float("inf") or grad_norm != grad_norm:
        #     #             self.overflow = True
                        
        #     self._finalize_grads(_exp_id)
            
        #     with nvtx.annotate("Step"):
        #         if not self.overflow:
        #             self.optimizers[_exp_id].step()
                
        #     self.zero_grad(_exp_id)
        #     self.set_grad_to_none(_exp_id)
        
        def _step(_exp_id, _stream):
            
            # if self.global_step < 10:
            #     grad_norm = self._clip_grad_norm_experts(_exp_id)
                    
            _stream.synchronize()
            
            ## NOTE
            ## maybe no need to call finalize grads since on the optimze process,
            ## the cpu_grad is already defined as grad
            ## So, if we fill the values for the cpu_grad, it will be ready to use.
            
            # if self.global_step < 10:
            #     with nvtx.annotate('overflow check'):
            #         if grad_norm == float("inf") or grad_norm != grad_norm:
            #             self.overflow = True
        
            # self._finalize_grads(_exp_id)
            
            ## Put exp_id to siganl that stream has been synchronized
            if not self.overflow:
                self.step_queue.put((Optim_TaskType.STEP, (_exp_id, False)))
                # print(f'main process {exp_id} has been queued')
            else:
                self.step_queue.put((Optim_TaskType.STEP, (_exp_id, True)))
            
            ## Make the step event set() to wake up the optim process
            self.optim_event.set()
            # print('main process optim event set')
            ## TODO
            ## Main process will be in sleep mode if I call this function
            ## Maybe call this function on wait previous optim
            ## But if I do that, the thread will be ended as soon as the stream is synchronized.
            # self.optim_expert_events[_exp_id].wait()
            
            # print(f'main process optim expert {_exp_id} event wait finished')
            ## NOTE
            ## After the step process, we need to make the grad field to None
            ## It could be placed right after stream sycnrhonize. 
            ## We don't need grad field anymore after DtoH copy.
            # self.set_grad_to_none(_exp_id)
            
            # print(f'main process {_exp_id} thread done')
            
        thread = Thread(target=_step, args=(exp_id, stream, ))
        thread.start()
        self.thread_list[exp_id] = thread