#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>

// CUDA runtime includes
#include <cuda_runtime_api.h>

// CUDA utilities and system includes
// #include <helper_cuda.h>
// #include <cuda.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <semaphore.h>
#include <fcntl.h>

#define HANDLE_KEY_NUM 960502

typedef struct ipcCUDA_st
{
    cudaIpcEventHandle_t eventHandle;
    cudaIpcMemHandle_t memHandle;
} ipcCUDA_t;


ipcCUDA_t* get_shared_handle(int *shmid, int rank){
    
    key_t key = (key_t) (HANDLE_KEY_NUM + rank);
    void *memory_segment=NULL;

    ipcCUDA_t *mem;

    if((*shmid = shmget(key, sizeof(ipcCUDA_t), IPC_CREAT|0666)) == -1){
        std::cout << "shmget failed\n"<< std::endl;
        exit(0);
    }

    if((memory_segment = shmat(*shmid, NULL, 0)) == (void*)-1){
        std::cout << "shmat failed\n"<< std::endl;
        exit(0);
    }
    return (ipcCUDA_t *) memory_segment;
}

#define MASTER_SEM_NAME "/mastersems"
#define SLAVE_SEM_NAME "/slavesems"
struct Tensor_Metadata {
    torch::Tensor tensor;
    int rank;

    Tensor_Metadata(torch::Tensor t, int r) : tensor(t), rank(r) {}
};

void slave_share_device_ptr(void* userData){
    auto metadata = (Tensor_Metadata *) userData;
    // Use Arrow

    int handle_id;
    ipcCUDA_t *s_mem;
    s_mem = get_shared_handle(&handle_id, metadata->rank);
    std::cout << "slave get shared handle" <<std::endl;

    sem_t* master_sem = sem_open(MASTER_SEM_NAME, O_CREAT, 0644, 0);
    sem_t* slave_sem = sem_open(SLAVE_SEM_NAME, O_CREAT, 0644, 0);

    if (master_sem == SEM_FAILED) {
        std::cerr << "Error opening slave master semaphore" << std::endl;
    }
    if (slave_sem == SEM_FAILED) {
        std::cerr << "Error opening slave slave semaphore" << std::endl;
    }

    std::cout << "slave get semaphores" <<std::endl;

    auto d_ptr = (metadata->tensor).data_ptr<float>();
   
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &(*s_mem).memHandle, (void *) d_ptr));
    
    // std::cout << "Slave's tensor d_ptr: " << (void *) d_ptr << std::endl;
    // std::cout << "Slave's tensor after IPC d_ptr: " << (void *) d_ptr << std::endl;
    //std::cout<< "Slave give through IPC" << std::endl;

    // b.1: wait until all event handles are created in other processes
    // procBarrier(s_lock);
    std::cout << "slave signal slave semaphore" <<std::endl;
    sem_post(slave_sem);
    std::cout << "slave wait for master semaphore" <<std::endl;
    sem_wait(master_sem);
    std::cout << "slave woke up" <<std::endl;
    //std::cout<< "Slave first barrier" << std::endl;

    C10_CUDA_CHECK(cudaIpcOpenEventHandle(&event, (*s_mem).eventHandle));

    // b.2: wait until all kernels launched and events recorded
    // sleep(1);
    // procBarrier(s_lock);
    // initBarrier(s_lock, 2);
    std::cout << "slave wait for master semaphore" <<std::endl;
    sem_wait(master_sem);
    std::cout << "slave woke up" <<std::endl;

    C10_CUDA_CHECK(cudaEventSynchronize(event));
    
    // b.2: Signal to Master that the event has been synchronized.
    // procBarrier(s_lock);
    std::cout << "slave signal slave semaphore" <<std::endl;
    sem_post(slave_sem);
    
    // Close the shared IPCCuda
    if(shmdt(s_mem) == -1){
        std::cout<< "shmdt failed" << std::endl;
    }

    if (sem_close(master_sem) != 0) {
        std::cerr << "Error closing master semaphore" << std::endl;
        return;
    }

    if (sem_close(slave_sem) != 0) {
        std::cerr << "Error closing slave semaphore" << std::endl;
        return;
    }

    return;
}

void master_get_device_ptr(void* userData){
    auto metadata = (Tensor_Metadata *) userData;

    int handle_id;
    ipcCUDA_t *s_mem;
    s_mem = get_shared_handle(&handle_id, metadata->rank);

    std::cout << "master get shared handle" <<std::endl;
    sem_t* master_sem = sem_open(MASTER_SEM_NAME, O_CREAT, 0644, 0);
    sem_t* slave_sem = sem_open(SLAVE_SEM_NAME, O_CREAT, 0644, 0);
    std::cout << "master get semaphores" <<std::endl;
    if (master_sem == SEM_FAILED) {
        std::cerr << "Error opening master master semaphore" << std::endl;
    }
    if (slave_sem == SEM_FAILED) {
        std::cerr << "Error opening master slave semaphore" << std::endl;
    }


    void* d_ptr;
    cudaEvent_t event;
    C10_CUDA_CHECK(cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess););
    C10_CUDA_CHECK(cudaIpcGetEventHandle((cudaIpcEventHandle_t *) &(*s_mem).eventHandle, event));

    // std::cout<< "master sempost master_sem" << master_sem << std::endl;
    // b.1: Signal to Slave Process that IPC event is created
    std::cout << "master signal master semaphore" <<std::endl;
    sem_post(master_sem);
    std::cout << "master wait slave semaphore" <<std::endl;
    sem_wait(slave_sem);
    std::cout << "master woke up" <<std::endl;
    // std::cout<< "Master first d_ptr " << d_ptr << std::endl;

    C10_CUDA_CHECK(cudaIpcOpenMemHandle((void **) &d_ptr, (*s_mem).memHandle, cudaIpcMemLazyEnablePeerAccess));
    auto dims = metadata->tensor.sizes();
    auto slave_gpu_tensor = torch::from_blob(d_ptr, dims, torch::TensorOptions().device(torch::kCUDA, metadata->rank));
    slave_gpu_tensor.copy_((metadata->tensor).to(torch::Device(torch::kCUDA, metadata->rank), true));
    C10_CUDA_CHECK(cudaEventRecord(event));

    //checkCudaErrors(cudaMemcpy((void *) d_ptr, (void *) host_tensor_ptr, DATA_BUF_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    //std::cout<< "Master get through IPC" << std::endl;
    //std::cout << "Master's original d_ptr: " <<  reinterpret_cast<void *>(host_pointer)  << std::endl;
    //std::cout << "Master's tensor d_ptr: " << host_tensor.data_ptr<float>() << std::endl;
    // std::cout << "Master slave's d_ptr: " << (void *)  d_ptr << std::endl;
    //torch::Tensor gpu_master_tensor = host_tensor.to(torch::Device(torch::kCUDA, 0), true);
    //std::cout<< "Master gpu tensor: "<< gpu_master_tensor << std::endl;
    //torch::IntArrayRef{host_tensor.size(0), host_tensor.size(1)}; //torch::IntArrayRef{1, 10, 512, 512};
    //std::cout<< "Master dims: "<< dims << std::endl;
    // int a = 3;
    // auto size = gpu_slave_tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(gpu_slave_tensor.dtype()));
    // std::cout << "Tensor Size: " << size << " " << sizeof(a) << " " << sizeof(gpu_slave_tensor) << std::endl;
    //std::cout<< "Master slave tensor:  "<< gpu_slave_tensor << std::endl;
    //std::cout<< "Master tensor pointer:  "<< gpu_slave_tensor.data_ptr<float>() << d_ptr << std::endl;
    //std::cout<< "Master final tensor:  "<< gpu_slave_tensor << std::endl;
    
    // b.2
    // procBarrier(s_lock);
    std::cout << "master signal master semaphore" <<std::endl;
    sem_post(master_sem);
    C10_CUDA_CHECK(cudaIpcCloseMemHandle(d_ptr));
    
    // b.3 wait till all the events are used up
    // procBarrier(s_lock);
    std::cout << "master wait slave semaphore" <<std::endl;
    sem_wait(slave_sem);
    std::cout << "master woke up" <<std::endl;
    C10_CUDA_CHECK(cudaEventDestroy(event));

    if(shmdt(s_mem) == -1){
        std::cout<< "shmdt failed" << std::endl;
    }

    if (sem_close(master_sem) != 0) {
        std::cerr << "Error closing master semaphore" << std::endl;
        return;
    }

    if (sem_close(slave_sem) != 0) {
        std::cerr << "Error closing slave semaphore" << std::endl;
        return;
    }

    return;
}

void cudaEnqueue(torch::Tensor tensor, int from_rank, int to_rank){
    Tensor_Metadata* userData = new Tensor_Metadata(tensor, to_rank);
    if (from_rank != to_rank){
        // cudaHostFn_t fn = master_get_device_ptr;
        // C10_CUDA_CHECK(cudaLaunchHostFunc(0, fn, (void *) userData));
        master_get_device_ptr((void *) userData);
    } 
    else {
        // cudaHostFn_t fn = slave_share_device_ptr;
        // C10_CUDA_CHECK(cudaLaunchHostFunc(0, fn, (void *) userData));
        slave_share_device_ptr((void *) userData);
    }
}