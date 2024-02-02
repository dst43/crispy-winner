#include <iostream>
#include <torch/extension.h>

int get_shared_memory(
    int rank, 
    torch::Tensor dev_tensor, 
    torch::Tensor host_tensor,
    unsigned long long dev_pointer,
    unsigned long long host_pointer
    );

int test(
    int rank, 
    torch::Tensor dev_tensor, 
    torch::Tensor host_tensor,
    unsigned long long dev_pointer,
    unsigned long long host_pointer) {
    
    std::cout << "test function fired" << std::endl;

    return get_shared_memory(rank, dev_tensor, host_tensor, dev_pointer, host_pointer);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("shared_memory", &test, "Get Tensor from pointer");
}