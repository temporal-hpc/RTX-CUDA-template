#include <iostream>
#include <iomanip>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <thrust/device_vector.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include <cuda.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#define BSIZE 1024
#define WARPSIZE 32
#define RTX_REPEATS 1
#define ALG_CLASSIC         0
#define ALG_WARP_SHUFFLE    1
#define ALG_CUB             2
#define ALG_THRUST          3
#define ALG_RTX_CLOSEST_HIT 4
const char *algStr[6] = {"", "WARP_SHUFFLE", "CUB", "THRUST", "RTX_CLOSEST_HIT"};

#include "common/common.h"
#include "common/Timer.h"
#include "src/rand.cuh"
#include "src/tools.h"
#include "src/device_tools.cuh"
#include "src/device_simulation.cuh"
#include "src/cuda_methods.cuh"
#include "rtx_params.h"
#include "src/rtx_functions.h"
#include "src/rtx.h"


int main(int argc, char *argv[]) {
    printf("----------------------------------\n");
    printf("  RTX-CUDA Template by Temporal   \n");
    printf("----------------------------------\n");
    if(!check_parameters(argc)){
        exit(EXIT_FAILURE);
    }
    int dev = atoi(argv[1]);
    int n = atoi(argv[2]);
    //int k = atoi(argv[3]);
    int steps = atoi(argv[3]);
    int alg = atoi(argv[4]);
    printf("Params: dev=%i  n=%i  steps=%i  alg=%s (BSIZE=%i)\n\n", dev, n, steps, algStr[alg], BSIZE);
    cudaSetDevice(dev);
    print_gpu_specs(dev);
    // 1) data on GPU, result has the resulting array and the states array
    std::pair<float*, curandState*> p = create_random_array_dev(n, 1123);

    // 2) computation
    switch(alg){
        case ALG_WARP_SHUFFLE:
            cudaWarpShuffle(n, steps, p.first, p.second);
            break;
        case ALG_CUB:
            cudaCUB(n, steps, p.first, p.second);
            break;
        case ALG_THRUST:
            cudaThrust(n, steps, p.first, p.second);
            break;
        case ALG_RTX_CLOSEST_HIT:
            rtx(n, 1, steps, alg, p.first, p.second);
            break;
    }
    printf("Benchmark Finished\n");
    return 0;
}
