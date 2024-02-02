import argparse
import torch
import sys
sys.path.append('../')
from torch.optim import Adam
from DeepSpeedAdam import DeepSpeedCPUAdam
from utils import AverageMeter
from fairseq.modules.transformer_layer import FeedForwardNetwork
import time
from threading import Thread

def main(args):

    num_networks = 16
    network_list = []
    
    for _ in range(num_networks):
        network_list.append(FeedForwardNetwork(args, embed_dim=args.embed_dim, ffn_dim=args.ffn_dim))
    
    torch_adam_list = []
    deepspeed_adam_list = []
    for i in range(num_networks):
        torch_adam_list.append(Adam(network_list[i].parameters(), lr=0.0001))
        deepspeed_adam_list.append(DeepSpeedCPUAdam(network_list[i].parameters(), adamw_mode=False))

    torch_average_meter = AverageMeter('torch_cpu_adam_meter')
    deepspeed_average_meter = AverageMeter('deepspeed_cpu_adam_meter')
    
    for network in network_list:
        for param in network.parameters():
            param.grad = torch.ones_like(param)

    for i in range(10):
        thread_list = []
        cpu_start = time.time() * 1000 
        for j in range(num_networks):
            
            def _step(k):
                torch_adam_list[k].step()
            
            thread = Thread(target=_step, args=(j, ))
            thread_list.append(thread)
            thread.start()
            
        for thread in thread_list:
            thread.join()
        cpu_end = time.time() * 1000
        torch_average_meter.update((cpu_end - cpu_start))
    
    for i in range(10):
        thread_list = []
        cpu_start = time.time() * 1000 
        for j in range(num_networks):
            
            def _step(k):
                deepspeed_adam_list[k].step()
            
            thread = Thread(target=_step, args=(j, ))
            thread_list.append(thread)
            thread.start()
            
        for thread in thread_list:
            thread.join()
        cpu_end = time.time() * 1000
        deepspeed_average_meter.update((cpu_end - cpu_start))
        

    print(torch_average_meter)
    print(deepspeed_average_meter)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    ## Training Args ##
    parser.add_argument('--embed-dim', default=2048, type=int)
    parser.add_argument('--seq-length', default=2048, type=int)
    parser.add_argument('--batch-size', default = 2, type=int)
    
    parser.add_argument('--expert-capacity', default=1.0, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--moe-expert-count', default=4, type=int)
    parser.add_argument('--iterations', default=10, type=int)
    parser.add_argument('--opt-type', default='torch', type=str)
    parser.add_argument('--update-freq', default=1, type=int)
    
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
    args.tokens_per_sample = args.seq_length

    main(args)