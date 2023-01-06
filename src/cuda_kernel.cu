#include "common/common.h"

__global__ void myKernel(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}

void launch_cuda(int n, float a, float *x, float *y) {
  myKernel<<<(n + 255) / 256, 256>>>(n, a, x, y);
  CUDA_CHECK(cudaGetLastError());
}
