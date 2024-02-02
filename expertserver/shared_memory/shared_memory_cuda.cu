#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <semaphore.h>
// CUDA runtime includes
#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <atomic>
// CUDA utilities and system includes
// #include <helper_cuda.h>
// #include <cuda.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <semaphore.h>

#define HANDLE_KEY_NUM 960502
#define CV_KEY_NUM 970808

typedef struct ipcCUDA_st
{
    int device;
    pid_t pid;
    cudaIpcEventHandle_t eventHandle;
    cudaIpcMemHandle_t memHandle;
} ipcCUDA_t;

// typedef struct simple_Lock
// {
//     bool init;
//     int number;
// }simple_Lock;

ipcCUDA_t* get_shared_handle(int *shmid){
    
    key_t key = (key_t) HANDLE_KEY_NUM;
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

// void initBarrier(simple_Lock* lock, int target){
//     if (lock->init)
//         return;
//     //std::cout<< "!!!!!!!!!!!!!!!!!! INIT !!!!!!!!!!!!!!" << std::endl;
//     lock->init = true;
//     lock->number = target;
// }


// void procBarrier(simple_Lock* lock){
//     lock->number--;
//     //std::cout<< "lock changed"  << lock->number << std::endl;
//     while(lock->number > 0){
//         continue;
//     };
// }

typedef struct CV
{
    std::mutex mutex;
    std::condition_variable cond;
    std::atomic<bool> slave{false};
    std::atomic<bool> master{false};

} CV;

CV* get_shared_cv(int *shmid){
    key_t key = (key_t) CV_KEY_NUM;
    void *memory_segment=NULL;

    CV* cv;
    std::cout << sizeof(CV) << " " << sizeof(*cv) <<std::endl;
    if((*shmid = shmget(key, sizeof(CV), IPC_CREAT|0666)) == -1){
        std::cout << "shmget failed\n"<< std::endl;
        exit(0);
    }

    if((memory_segment = shmat(*shmid, NULL, 0)) == (void*)-1){
        std::cout << "shmat failed\n"<< std::endl;
        exit(0);
    }

    return (CV *) memory_segment;
}

// void initCV() {
    
//     int cv_id;
//     CV *s_cv;
//     s_cv = get_shared_cv(&cv_id);

//     pthread_mutex_init(&s_cv->mutex, NULL);
//     pthread_cond_init(&s_cv->cond, NULL);

//     if(shmdt(s_cv) == -1){
//         std::cout<< "shmdt failed" << std::endl;
//     }

//     return;
// }

void get_shared_memory(int rank, torch::Tensor dev_tensor, torch::Tensor host_tensor, unsigned long long dev_pointer, unsigned long long host_pointer){
    
    //cudaLaunchHostFunc();
    //std::cout<< rank << "get shared_memory fired" << std::endl;

    int handle_id;
    ipcCUDA_t *s_mem;
    s_mem = get_shared_handle(&handle_id);
    
    //std::cout<< rank << "get shared handle" << std::endl;
    // if (rank == 99){
    //     initCV();
    // }
    
    // int cv_id;
    // CV* s_cv;
    // s_cv = get_shared_cv(&cv_id);
    // initBarrier(s_lock, 2);

    //std::cout<< rank << "get shared CV" << std::endl;

    //Slave Process rank [0, 99)

    if (rank == 0){
        if(shmctl(handle_id, IPC_RMID, NULL) == -1){
            printf("shmctl failed\n");
        }

        // if(shmctl(cv_id, IPC_RMID, NULL) == -1){
        //     printf("shmctl failed\n");
        // }

        s_mem = get_shared_handle(&handle_id);
        // s_cv = get_shared_cv(&cv_id);
    }

    const char* SEM_NAME = "/my_semaphore";
    sem_t* semaphore = sem_open(SEM_NAME, O_CREAT, 0644, 0);

    std::cout << "Semaphore initialization finished " << semaphore << std::endl;

    if (semaphore == SEM_FAILED) {
        perror("sem_open(3) error");
        exit(EXIT_FAILURE);
    }

    if (rank != 99){
        auto d_ptr = dev_tensor.data_ptr<float>();

        std::cout << "Slave's original d_ptr: " <<  reinterpret_cast<void *>(dev_pointer) <<std::endl;
        std::cout << "Slave's tensor d_ptr: " << (void *) d_ptr << std::endl;
        
        cudaEvent_t event;

        C10_CUDA_CHECK(cudaIpcGetMemHandle((cudaIpcMemHandle_t *) &(*s_mem).memHandle, (void *) d_ptr));
        
        // s_cv->mutex.lock();
        
        // s_cv->mutex.unlock();
        std::cout << "Slave's tensor after IPC d_ptr: " << (void *) d_ptr << std::endl;

        //std::cout<< "Slave give through IPC" << std::endl;

        // b.1: wait until all event handles are created in other processes
        // procBarrier(s_lock);

        std::cout << "Slave process waiting for semaphore" << std::endl;
        sem_wait(semaphore);
        std::cout << "Slave process woke up" << std::endl;

        // std::cout << "Slave process signaling semaphore" << std::endl;
        // sem_post(semaphore);

        //std::cout<< "Slave first barrier" << std::endl;
        // std::unique_lock<std::mutex> lk(s_cv->mutex);
        // std::cout << "Slave get unique lock" << std::endl;
        // s_cv->master = true;
        // s_cv->cond.notify_one();
        // std::cout << "Slave notify" << std::endl;
        
        // // s_cv->cond.wait(lk, [&]{return s_cv->slave == true;});
        // while(!s_cv->slave){
        //     continue;
        // }
        // s_cv->slave = false;
        // lk.unlock();

        C10_CUDA_CHECK(cudaIpcOpenEventHandle(&event, (*s_mem).eventHandle));

        std::cout<< "Slave Event Handle" << std::endl;
        // b.2: wait until all kernels launched and events recorded

        // sem_post(semaphore);
        // sem_wait(semaphore);

        std::cout << "Slave process waiting for semaphore" << std::endl;
        sem_wait(semaphore);
        std::cout << "Slave process woke up" << std::endl;

        // sleep(1);
        // procBarrier(s_lock);
        // initBarrier(s_lock, 2);
        // s_cv->master = true;
        // s_cv->cond.notify_one();
        // // std::unique_lock<std::mutex> lk2(s_cv->mutex);
        // s_cv->cond.wait(lk, [&]{return s_cv->slave == true;});
        // s_cv->slave = false;
        // lk2.unlock();

        C10_CUDA_CHECK(cudaEventSynchronize(event));

        std::cout<< "Slave event synchronize" << std::endl;

        std::cout << "Slave process signaling semaphore" << std::endl;
        sem_post(semaphore);

        // procBarrier(s_lock);
        //C10_CUDA_CHECK(cudaDeviceSynchronize());
    }
    //Master rank = 99
    else {
        void* d_ptr;
       
        cudaEvent_t event;
        C10_CUDA_CHECK(cudaEventCreate(&event, cudaEventDisableTiming | cudaEventInterprocess));
        C10_CUDA_CHECK(cudaIpcGetEventHandle((cudaIpcEventHandle_t *) &(*s_mem).eventHandle, event));

        //std::cout<< "create event handle" << std::endl;

        // b.1: wait until proc 0 initializes device memory
        // procBarrier(s_lock);
        // initBarrier(s_lock, 2);
        
        std::cout << "Master process signaling semaphore" << std::endl;
        sem_post(semaphore);

        // std::cout << "Master process waiting for semaphore" << std::endl;
        // sem_wait(semaphore);
        // std::cout << "Master process woke up" << std::endl;

        // std::unique_lock<std::mutex> lk(s_cv->mutex);
        // std::cout << "Master get unique lock" << std::endl;
        // s_cv->cond.wait(lk, [&]{return s_cv->master == true;});
        // std::cout << "Master cond wait" << std::endl;
        // s_cv->master = false;
        // lk.unlock();

        std::cout<< "Master first d_ptr " << d_ptr << std::endl;

        C10_CUDA_CHECK(cudaIpcOpenMemHandle((void **) &d_ptr, (*s_mem).memHandle, cudaIpcMemLazyEnablePeerAccess));
        //checkCudaErrors(cudaMemcpy((void *) d_ptr, (void *) host_tensor_ptr, DATA_BUF_SIZE * sizeof(int), cudaMemcpyHostToDevice));
        
        //std::cout<< "Master get through IPC" << std::endl;
        std::cout << "Master's original d_ptr: " <<  reinterpret_cast<void *>(host_pointer)  << std::endl;
        std::cout << "Master's tensor d_ptr: " << host_tensor.data_ptr<float>() << std::endl;
        std::cout << "Master slave's d_ptr: " << (void *)  d_ptr << std::endl;
        //torch::Tensor gpu_master_tensor = host_tensor.to(torch::Device(torch::kCUDA, 0), true);
        //std::cout<< "Master gpu tensor: "<< gpu_master_tensor << std::endl;
        auto dims = host_tensor.sizes(); //torch::IntArrayRef{host_tensor.size(0), host_tensor.size(1)}; //torch::IntArrayRef{1, 10, 512, 512};
        //std::cout<< "Master dims: "<< dims << std::endl;
        auto gpu_slave_tensor = torch::from_blob(d_ptr, dims, torch::TensorOptions().device(torch::kCUDA, 0));
        int a = 3;

        auto size = gpu_slave_tensor.numel() * torch::elementSize(torch::typeMetaToScalarType(gpu_slave_tensor.dtype()));
        std::cout << "Tensor Size: " << size << " " << sizeof(a) << " " << sizeof(gpu_slave_tensor) << std::endl;
        //std::cout<< "Master slave tensor:  "<< gpu_slave_tensor << std::endl;
        //std::cout<< "Master tensor pointer:  "<< gpu_slave_tensor.data_ptr<float>() << d_ptr << std::endl;
        gpu_slave_tensor.copy_(host_tensor);
        //std::cout<< "Master final tensor:  "<< gpu_slave_tensor << std::endl;
        
        C10_CUDA_CHECK(cudaEventRecord(event));
        
        // b.2
        // procBarrier(s_lock);

        std::cout << "Master process signaling semaphore" << std::endl;
        sem_post(semaphore);

        // std::cout << "Master process waiting for semaphore" << std::endl;
        // sem_wait(semaphore);
        // std::cout << "Master process woke up" << std::endl;

        // s_cv->slave = true;
        // s_cv->cond.notify_all();
        // std::cout << "Master notify" << std::endl;
        // sleep(1);
        // // std::unique_lock<std::mutex> lk2(s_cv->mutex);
        // std::cout << "Master get unique lock" << std::endl;
        // s_cv->cond.wait(lk, [&]{return s_cv->master == true;});
        // s_cv->master = false;
        // // lk2.unlock();
        
        C10_CUDA_CHECK(cudaIpcCloseMemHandle(d_ptr));
        
        // b.3 wait till all the events are used up
        // procBarrier(s_lock);

        std::cout << "Master process waiting for semaphore" << std::endl;
        sem_wait(semaphore);
        std::cout << "Master process woke up" << std::endl;

        // s_cv->slave = true;
        // s_cv->cond.notify_one();
        // // std::unique_lock<std::mutex> lk3(s_cv->mutex);
        // s_cv->cond.wait(lk, [&]{return s_cv->master == true;});
        // s_cv->master = false;
        // // lk3.unlock();
        
        // sem_post(semaphore);
        // sem_wait(semaphore);

        C10_CUDA_CHECK(cudaEventDestroy(event));
    }
}

