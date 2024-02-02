#include <torch/torch.h>
#include <sys/mman.h> // for mmap and munmap
#include <fcntl.h> // for open
#include <torch/extension.h>

torch::nn::AnyModule createNet() {
  // Load the state dictionary of the network from the file
  torch::OrderedDict<std::string, torch::Tensor> state_dict;
  torch::load(state_dict, "net.pth");

  // Create an instance of the neural network architecture using the loaded state dictionary
  Net net(10, 20, 1);
  net.load_state_dict(state_dict);

  // Share the network across multiple processes
  torch::DistributedC10d::init_process_group();

  // Get the tensor pointing to the parameters of the network
  std::vector<torch::Tensor> params;
  for (auto& p : net.parameters()) {
    params.push_back(p);
  }
  torch::Tensor flat_params = torch::cat(torch::flatten(params));

  // Pin the tensor
  void* ptr = mmap(nullptr, flat_params.numel() * sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    throw std::runtime_error("Failed to allocate shared memory");
  }
  memcpy(ptr, flat_params.data_ptr(), flat_params.numel() * sizeof(float));
  flat_params = torch::from_blob(ptr, flat_params.sizes(), flat_params.strides(), torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
  flat_params = flat_params.pin_memory();

  // Create a new module from the network
  torch::nn::AnyModule any_net(net);

  // Add the pinned tensor to the module's buffers
  any_net.register_buffer("params", flat_params);

  // Return the module
  return any_net;
}

PYBIND11_MODULE(example, m) {
  m.def("createNet", &createNet, "Create a shared and pinned neural network.");
}