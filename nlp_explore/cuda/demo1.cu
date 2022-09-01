
__global__ void TwoPassWarpSyncKernel(const float* input, float* part_sum, size_t n) {
    int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total_thread_num = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int32_t i = gtid; i<n; i+=total_thread_num){
        sum += input[i];
    }

    // store sum to shared memory
    extern __shared__ float shm[];
    shm[threadIdx.x] = sum;
    __syncthreads();

    for (int32_t active_thread_num=blockDim.x/2; active_thread_num>32; active_thread_num /= 2) {
        if (threadIdx.x < active_thread_num) {
            shm[threadIdx.x] += shm[threadIdx.x + active_thread_num]
        }
        __syncthreads();
    }

    //final warp
    if (threadIdx.x < 32) {
        violatile float* vshm = shm;     //使用warp隐式同步时使用shared memory需要配合volatile关键字
        if (blockDim.x >= 64) {
            vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        }
        vshm[threadIdx.x] += vshm[threadIdx.x + 16];
        vshm[threadIdx.x] += vshm[threadIdx.x + 8];
        vshm[threadIdx.x] += vshm[threadIdx.x + 4];
        vshm[threadIdx.x] += vshm[threadIdx.x + 2];
        vshm[threadIdx.x] += vshm[threadIdx.x + 1];
        if (threadIdx.x == 0) {
            part_sum[blockIdx.x] = vshm[0]
        }
    }
}
