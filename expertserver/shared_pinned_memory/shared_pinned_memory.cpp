#include <iostream>
#include <torch/extension.h>

torch::Tensor shared_pinned_memory(
    torch::Tensor tensor,
    int rank,
    int layer, 
    int expert, 
    int order,
    int type,
    bool pinning = true
    );

torch::Tensor shared_pinned_memory_func(
    torch::Tensor tensor,
    int rank,
    int layer, 
    int expert, 
    int order,
    int type,
    bool pinning = true) {

    return shared_pinned_memory(tensor, rank, layer, expert, order, type, pinning);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_pinned_memory", &shared_pinned_memory_func, "Get Shared and Pinned Tensor");
}