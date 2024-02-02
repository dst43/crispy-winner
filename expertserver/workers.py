
import torch
from threading import Thread
from queue import Empty
from typing import Optional
from utils import Communicator, TaskType

from experts_cpuadam import Experts_CPUAdam
from experts_cpu import Experts


class CPUWorker(object):
    def __init__(self, args):
        self.thread: Optional[Thread] = None
        self.args = args
        self.manager = torch.multiprocessing.Manager()
        self.comm = Communicator(self.manager, args.ngpu, args.moe_expert_count)
        self.subworkers = [SubCPUWorker(self, self.comm, dev_id) for dev_id in range(args.ngpu)]
        
        self._init_experts()
        
    def _init_experts(self):
        experts = Experts(self.args, self.args.moe_expert_count)
        experts_optimizer = Experts_CPUAdam(experts.expert_list, self.args.opt_type, lr = 0.1)
        experts.set_optimizer(experts_optimizer)
        self.experts = experts
        
    def start(self):
        self.thread = Thread(target=self._thread, args=())
        self.thread.name = f'CPU Master thread'
        self.thread.start()
    
    def _thread(self):
        self.subworker_threads = []
        for subworker in self.subworkers:
            self.subworker_threads.append(subworker.start())
        
    def stop(self):    
        for sub_threads in self.subworker_threads:
            sub_threads.join()

        
class SubCPUWorker(object):
    def __init__(self, baseworker:CPUWorker, comm, dev_id):
        self.baseworker = baseworker
        self.comm = comm
        
        self.signal_queue, self.model_queue, self.grad_queue = comm.getQueue(dev_id)
        self.grad_count = 4
        
        self.dev_id = dev_id
        self.thread: Optional[Thread] = None
        
    def start(self):
        self.thread = Thread(target=self._thread, args=())
        self.thread.name = f'CPU{self.dev_id} thread'
        self.running = True
        print(f'subworker{self.dev_id} thread start call')
        self.thread.start()
        return self.thread
        
    def _thread(self):
        print(f'subworker{self.dev_id} thread started')
        while True:
            if not self.running:
                print(f"Terminating thread {self.thread.name}")
                return
            
            try:
                metadata = self.signal_queue.get()
                #print('metadata')
            except Empty:
                continue
            #print(f'metadata received:{metadata}')
            exp_id, task = metadata[0], metadata[1]
            
            if task == TaskType.TASK_PARAM:
                def _param_thread():
                    expert_params = list(self.baseworker.experts.expert_list[exp_id].parameters())
                    self.model_queue.put(expert_params)
                param_thread = Thread(target=_param_thread)
                param_thread.start()
            elif task == TaskType.TASK_GRAD_AND_OPTIM:
                continue
                ## Grads are expected to be put in the queue individually, so we need to gather all.
                next_expert_grads = []
                ## TODO
                ## change this to count
                while self.grad_count != 0: #self.model_queue.qsize() != 0:
                    next_expert_grads.append(self.grad_queue.get())
                    self.grad_count -= 1
                    
                expert_params = list(self.baseworker.experts.expert_list[exp_id].parameters())
                assert len(expert_params) == len(next_expert_grads)

                ## Change the attributes of the current grad to new received grads.
                ## Also, since we get the gradients from the reverse direction of parameter, we reverse it
                for param, new_grad in zip(reversed(expert_params), next_expert_grads):
                    print(param.shape, new_grad.shape)
                    #param.grad = new_grad
                
                ## OPTIMIZE
                self.grad_count = 4
                self.baseworker.experts.optimizer.step(exp_id)
                self.comm.optimize_count_increase(exp_id)
                
            elif task == TaskType.TASK_TERMINATE:
                self.stop()
            else:
                print(f'not implmented task: {metadata}')
                self.stop()
            
    def stop(self):
        self.running = False
            
            