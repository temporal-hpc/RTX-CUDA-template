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

    CmdArgs args = get_args(argc, argv);
    int dev = args.dev;
    int n = args.n;
    //int k = atoi(argv[3]);
    int steps = args.steps;
    int alg = args.alg;
    int seed = 1123;

    cudaSetDevice(dev);
    print_gpu_specs(dev);
    // 1) data on GPU, result has the resulting array and the states array
    curandState* devStates = setup_curand(n, seed);
    float* d_array = create_random_array_dev<float>(n, 100.0, devStates);


    // 2) computation
    switch(alg){
        case ALG_WARP_SHUFFLE:
            cudaWarpShuffle(n, steps, d_array, devStates);
            break;
        case ALG_CUB:
            cudaCUB(n, steps, d_array, devStates);
            break;
        case ALG_THRUST:
            cudaThrust(n, steps, d_array, devStates);
            break;
        case ALG_RTX_CLOSEST_HIT:
            rtx(n, 1, steps, alg, d_array, devStates);
            break;
    }
    printf("Benchmark Finished\n");
    return 0;
}
