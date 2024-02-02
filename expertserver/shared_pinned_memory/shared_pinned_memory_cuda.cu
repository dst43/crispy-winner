// CUDA runtime includes
#include <cuda_runtime_api.h>

// CUDA utilities and system includes
// #include <helper_cuda.h>
// #include <cuda.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <sys/mman.h>
#include <fcntl.h>
#include <sys/shm.h>

#define KEY_NUM 1000000

torch::Tensor shared_pinned_memory(torch::Tensor tensor, int rank, int layer, int expert, int order, int type, bool pinning = true){
    int shm_id;
    auto key_val = KEY_NUM * type;
    key_t key = (key_t) (key_val + layer * 10000 + expert * 10 + order);

    void *memory_segment=NULL;

    auto tensor_size = tensor.numel() * tensor.element_size();

    if((shm_id = shmget(key, tensor_size, IPC_CREAT|0666)) == -1){
        std::cout << "shmget failed\n"<< std::endl;
        exit(0);
    }

    if((memory_segment = shmat(shm_id, NULL, 0)) == (void*)-1){
        std::cout << "shmat failed\n"<< std::endl;
        exit(0);
    }
    
    // std::cout <<"rank: " << rank << ", key: " << key << ", memory_segment: " << memory_segment << std::endl;

    // NOTE: Only Params and Grads should be pinned
    if (pinning && (type == 1 || type == 2)){
        C10_CUDA_CHECK(cudaHostRegister((void *) memory_segment, (size_t) tensor_size, cudaHostRegisterMapped));
    }
    
    auto shared_tensor = torch::from_blob(memory_segment, tensor.sizes());

    // NOTE: Only Params need initialization
    if ((rank == 0) && (type == 1)){
        shared_tensor.copy_(tensor);
    }
    // std::cout<< "Shared Tensor from blob" << std::endl;


    // Need to find a clever way of detaching it
    // if(shmctl(shm_id, IPC_RMID, NULL) == -1) {
    //     std::cerr<< "shmctl failed " << std::endl;
    // }

    return shared_tensor;
}