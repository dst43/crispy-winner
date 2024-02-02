from enum import Enum, IntEnum
import copy

global_layer_count = 0
global_cfg = {}
global_scaler = None
cpu_loader = None
global_lr = 0

def global_layer_count_increment():
    global global_layer_count
    global_layer_count += 1
    
def set_global_cfg(cfg):
    global global_cfg
    global_cfg = cfg
    
def set_global_scaler(scaler):
    global global_scaler
    global_scaler = scaler
    
def set_global_cpuloader(loader):
    global cpu_loader
    cpu_loader = loader

def set_global_lr(lr):
    global global_lr
    global_lr = lr

class TaskType(Enum):
    TASK_PARAM = 1
    TASK_GRAD_AND_OPTIM = 2
    TASK_TERMINATE = 3

class ParamType(IntEnum):
    PARAM = 1
    GRAD = 2
    OPTIM_EXP_AVG = 3
    OPTIM_EXP_AVG_SQ = 4
    OPTIM_STEP = 5

class Optim_TaskType(IntEnum):
    STEP = 1
    ACCUM = 2
    LR = 3
    JOIN = 4
    
class Communicator(object):
    def __init__(self, manager, ngpu, num_experts):
        self.signal_queue = [manager.Queue() for _ in range(ngpu)]
        self.model_queue = [manager.Queue() for _ in range(ngpu)]
        self.grad_queue = [manager.Queue() for _ in range(ngpu)]

        self.num_experts = num_experts
        self.optimize_cond = manager.Condition()
        self.optimize_count = [0 for _ in range(self.num_experts)]
        
    def getQueue(self, rank):
        return (self.signal_queue[rank], self.model_queue[rank], self.grad_queue[rank])

    def optimize_count_increase(self, exp_id):
        print('optimize_count fired')
        with self.optimize_cond:
            assert self.optimize_count[exp_id] == 0
            self.optimize_count[exp_id] = 1
            
            if sum(self.optimize_count) == self.num_experts:
                print('notify all workers')
                self.optimize_cond.notify_all()
                self.optimize_count = [0 for _ in range(self.num_experts)]
    
    def stop(self):
        metadata = [-1, TaskType.TASK_TERMINATE]
        for signal_queue in self.signal_queue:
            signal_queue.put(copy.deepcopy(metadata))
                
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
