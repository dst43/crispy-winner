import torch


if __name__ == '__main__':
    
    
    num_devices, num_experts = 4, 20
    
    torch.random.manual_seed(1)
    summed_list = torch.randint(low=750, high=1250, size=(num_experts,))
    capacity = int(summed_list.max().item())
    sorted_list, indices = summed_list.sort(descending = True)
    sorted_list, indices = sorted_list.tolist(), indices.tolist()
    bins, gpu_assign = [(0, 0) for _ in range(num_devices)], [[] for _ in range(num_devices)]
    max_assign = num_experts // num_devices
    for i in range(len(sorted_list)):
        idx_min = min(range(len(bins)), key=lambda idx: bins[idx][0] if bins[idx][1] < max_assign else float('inf'))
        val, count = bins[idx_min]
        bins[idx_min] = (val + sorted_list[i], count + 1)
        gpu_assign[idx_min].append(indices[i])
    print(summed_list)
    print(sorted_list, indices)
    for assign in gpu_assign:
        assign.sort()
    print(capacity, bins, gpu_assign)