#pragma once
#define NUM_REQUIRED_ARGS 5
void print_help(){
    fprintf(stderr, AC_BOLDGREEN "run as ./rtxcuda <dev> <n> <s> <alg>\n" AC_RESET
                    "dev = device ID\n"
                    "n   = problem size\n"
                    "s   = number of simulation steps\n"
                    "alg = algorithm\n"
                    "   1 -> %s\n"
                    "   2 -> %s\n"
                    "   3 -> %s\n"
                    "   4 -> %s\n"
                    "   5 -> %s\n",
                    algStr[1],
                    algStr[2],
                    algStr[3],
                    algStr[4],
                    algStr[5]);
}

bool check_parameters(int argc){
    if(argc != NUM_REQUIRED_ARGS){
        fprintf(stderr, AC_YELLOW "missing arguments\n" AC_RESET);
        print_help();
        return false;
    }
    return true;
}

void print_gpu_specs(int dev){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %d\n", prop.concurrentKernels);
    printf("  Memory Clock Rate (MHz):      %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits):      %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}
