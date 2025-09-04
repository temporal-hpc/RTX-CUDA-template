#pragma once

#include <unistd.h>
#include <string>
#include <time.h>
#include <getopt.h>
#include <iostream>
#include <fstream>

#define ARG_NB 2
#define ARG_REPS 3
#define ARG_DEV 4
#define ARG_NT 5
#define ARG_SEED 6
#define ARG_CHECK 7
#define ARG_TIME 8
#define ARG_POWER 9

struct CmdArgs {
    int n, alg, steps, reps, dev, nt, seed, check, save_time, save_power;
    std::string time_file, power_file;
};

struct Results {
    float output;
    float time;
    int mem;
    int power;
};

void print_help(){
    fprintf(stderr, AC_BOLDGREEN "run as ./rtxcuda <n> <s> <alg>\n" AC_RESET
                    "n   = problem size\n"
                    "s   = number of simulation steps\n"
                    "alg = algorithm\n"
                    "   1 -> %s\n"
                    "   2 -> %s\n"
                    "   3 -> %s\n"
                    "   4 -> %s\n"
                    "   5 -> %s\n"
                    "\n"
                    "Options:\n"
                    "   --reps <repetitions>      RMQ repeats for the avg time (default: 10)\n"
                    "   --dev <device ID>         device ID (default: 0)\n"
                    "   --nt  <thread num>        number of CPU threads\n"
                    "   --seed <seed>             seed for PRNG\n"
                    "   --check                   check correctness\n"
                    "   --save-time=<file>        \n"
                    "   --save-power=<file>       \n",
                    algStr[1],
                    algStr[2],
                    algStr[3],
                    algStr[4],
                    algStr[5]);
}

#define NUM_REQUIRED_ARGS 4
CmdArgs get_args(int argc, char *argv[]) {
    if (argc < NUM_REQUIRED_ARGS) {
        print_help();
        exit(EXIT_FAILURE);
    }

    CmdArgs args;
    args.n = atoi(argv[1]);
    args.steps = atoi(argv[2]);
    args.alg = atoi(argv[3]);
    if (!args.n || !args.steps) {
        print_help();
        exit(EXIT_FAILURE);
    }

    args.reps = 10;
    args.seed = time(0);
    args.dev = 0;
    args.check = 0;
    args.save_time = 0;
    args.save_power = 0;
    args.nt = 1;
    args.time_file = "";
    args.power_file = "";
    
    static struct option long_option[] = {
        // {name , has_arg, flag, val}
        {"reps", required_argument, 0, ARG_REPS},
        {"dev", required_argument, 0, ARG_DEV},
        {"nt", required_argument, 0, ARG_NT},
        {"seed", required_argument, 0, ARG_SEED},
        {"check", no_argument, 0, ARG_CHECK},
        {"save-time", optional_argument, 0, ARG_TIME},
        {"save-power", optional_argument, 0, ARG_POWER},
    };
    int opt, opt_idx;
    while ((opt = getopt_long(argc, argv, "12345", long_option, &opt_idx)) != -1) {
        if (isdigit(opt))
                continue;
        switch (opt) {
            case ARG_REPS:
                args.reps = atoi(optarg);
                break;
            case ARG_DEV:
                args.dev = atoi(optarg);
                break;
            case ARG_NT: 
                args.nt = atoi(optarg);
                break;
            case ARG_SEED:
                args.seed = atoi(optarg);
                break;
            case ARG_CHECK:
                args.check = 1;
                break;
            case ARG_TIME:
                args.save_time = 1;
                if (optarg != NULL)
                    args.time_file = optarg;
                break;
            case ARG_POWER:
                args.save_power = 1;
                if (optarg != NULL)
                    args.power_file = optarg;
                break;
            default:
                break;
        }
    }


    printf( "Params:\n"
            "   reps = %i\n"
            "   seed = %i\n"
            "   dev  = %i\n"
            AC_GREEN "   n     = %i (~%f GB, float)\n" AC_RESET
            //AC_GREEN "   q    = %i (~%f GB, int2)\n" AC_RESET
            AC_GREEN "   steps = %i\n" AC_RESET
            "   nt   = %i CPU threads\n"
            "   alg  = %i (%s)\n\n",
            args.reps, args.seed, args.dev, args.n, sizeof(float)*args.n/1e9, args.steps,
            args.nt, args.alg, algStr[args.alg]);

    return args;
}

bool is_equal(float a, float b) {
    float epsilon = 1e-4f;
    return abs(a - b) < epsilon;
}

bool check_parameters(int argc){
    if(argc != NUM_REQUIRED_ARGS){
        fprintf(stderr, AC_YELLOW "missing arguments\n" AC_RESET);
        print_help();
        return false;
    }
    return true;
}


void print_gpu_specs(int dev) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    // Use cudaDeviceGetAttribute to get deprecated properties
    int memoryClockRateKhz, memoryBusWidthBits;
    cudaDeviceGetAttribute(&memoryClockRateKhz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&memoryBusWidthBits, cudaDevAttrGlobalMemoryBusWidth, dev);

    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %d\n", prop.concurrentKernels);

    // Print the memory clock rate, converting kHz to MHz
    printf("  Memory Clock Rate (MHz):      %d\n", memoryClockRateKhz / 1000);
    // Print the memory bus width
    printf("  Memory Bus Width (bits):      %d\n", memoryBusWidthBits);

    // Calculate and print the peak memory bandwidth
    double peak_bandwidth = 2.0 * memoryClockRateKhz * memoryBusWidthBits / 8.0 / 1.0e6;
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", peak_bandwidth);
}


void write_results(CmdArgs args, Results results) {
    if (!args.save_time) return;
    std::string filename;
    if (args.time_file.empty())
        filename = std::string("../results/data.csv");
    else
        filename = args.time_file;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, args.dev);
    char *device = prop.name;
    //if (alg == ALG_CPU_BASE || alg == ALG_CPU_HRMQ) {
    //    strcpy(device, "CPU ");
    //    char hostname[50];
    //    gethostname(hostname, 50);
    //    strcat(device, hostname);
    //}

    FILE *fp;
    fp = fopen(filename.c_str(), "a");

    fprintf(fp, 
            "%s,%s,%i,%i," //args
            "%f\n", // results
            args.dev,
            algStr[args.alg],
            args.reps,
            args.n,
            results.time);
    fclose(fp);
}
