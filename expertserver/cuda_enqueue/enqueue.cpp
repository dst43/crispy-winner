#include <iostream>
#include <torch/extension.h>

void cudaEnqueue(
    torch::Tensor tensor, 
    int from_rank,
    int to_rank
    );

void cuda_enqueue_func(
    torch::Tensor tensor, 
    int from_rank,
    int to_rank){

    cudaEnqueue(tensor, from_rank, to_rank);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cuda_enqueue", &cuda_enqueue_func, "Get Tensor from pointer");
}