#pragma once
#include <curand_kernel.h>

__global__ void kernel_setup_prng(int n, int seed, curandState *state){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    if(id < n){
        curand_init(seed, id, 0, &state[id]);
    }
}

template <typename T>
__global__ void kernel_random_array(int n, T max, curandState *state, T *array){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if(id >= n){ return; }
    float x = curand_uniform(&state[id]);
    array[id] = x*max;
}

curandState* setup_curand(int n, int seed) {
    curandState *devStates;
    cudaMalloc((void **)&devStates, n * sizeof(curandState));

    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_setup_prng<<<grid, block>>>(n, seed, devStates);
    cudaDeviceSynchronize();

    return devStates;
}

template <typename T>
T* create_random_array_dev(int n, T max, curandState* devStates){
    T* darray;
    cudaMalloc(&darray, sizeof(T)*n);

    dim3 block(BSIZE, 1, 1);
    dim3 grid((n+BSIZE-1)/BSIZE, 1, 1); 
    kernel_random_array<<<grid,block>>>(n, max, devStates, darray);
    cudaDeviceSynchronize();

    return darray;
}



