import torch
from utils import AverageMeter
from experts_cpu import Experts
import nvtx

def main(args, comm):
    
    input_dim = int(args.batch_size * args.seq_length * args.expert_capacity)
    # ## lazy init should be distributed
    num_local_experts = args.moe_expert_count // args.world_size
    experts = Experts(args, num_local_experts, comm)
    experts.lazy_init()
    print('init finished')
    
    gpu_expert_assign = [[0, 1, 2, 3]]
    iterations = args.iterations
    
    throughput_meter = AverageMeter('training_speed')
    cuda_start = torch.cuda.Event(enable_timing=True)
    cuda_end = torch.cuda.Event(enable_timing=True)
    outputs = [torch.randn(input_dim, args.embed_dim, dtype=torch.float16).to('cuda') for _ in range(iterations)]
    for i in range(args.epochs):
        torch.cuda.synchronize()
        cuda_start.record()
        for it in range(iterations):
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
            #         print('equal: ')
            #         if p.grad is not None and p._cpu_grad is not None:
            #             print(p.grad is p._cpu_grad)
        
        
            ## Explicitly set grad to None since we don't have any params in experts but do have expert list
            output = outputs[it].chunk(num_local_experts, dim=0)
            
            ## Forward
            print('forward')
            with nvtx.annotate(f"forward{i}", color="blue"):
            
            # output = non_experts.pre_head(output)
                output = experts(output, gpu_expert_assign, rank = args.rank)
            #output = non_experts.post_head(output)
            
                output = torch.cat(output, dim = 0)

                loss = output.sum(dim = 0).sum(dim = 0)
                #torch.cuda.synchronize()

            ## Backward
            print('backward')
            with nvtx.annotate(f"backward{i}", color="red"):
                loss.backward()
                experts.set_grad_to_none()
                #torch.cuda.synchronize()
            #experts._wait_for_previous_optim_step()
            # print('optimize')
            ## Non-experts Optimize // Experts Optimize call happend in the post_gradient_hook
            # non_experts_optimizer.step()
            
            ## Non-experts Zero Grad // Experts Zero Grad called with step() Experts_CPUAdam.step(exp_id)
            # non_experts_optimizer.zero_grad()
        ## Finalize the Optimization Step
        # for t in experts_optimizer.thread_list:
        #     if t:
        #         t.join()
        torch.cuda.synchronize()
        cuda_end.record()
        if i > 0:
            throughput_meter.update(cuda_start.elapsed_time(cuda_end))
    
    print(throughput_meter)
    comm.stop()
    
def func_torch(rank, queue):
    
    temp = queue.get()
    print('rank:0', id(temp.storage()))