# RTX-CUDA-template
A simple RTX-CUDA interoperability test program based on the SAXPY example from NVIDIA's RTX Compute Samples https://github.com/NVIDIA/rtx_compute_samples

The sample code simulates `n` randomly generated numbers, by shifting their value (as if was a 1D coordinate) by a random translation, and computes the minimum value at each time step. The user can choose different approaches
- CUDA Kernel
- CUB
- Thrust
- RTX OptiX 

## Dependencies
- CUDA 11 or later
- OptiX 7.5 or later
- CUB (for comparison)
- THrust (for comparison)

## Compile and run
```
mkdir build && cd build
cmake ../ -DOPTIX_HOME=<PATH-TO-OPTIX-MAIN-DIR>
make
./rtx-cuda <dev> <n> <s> <alg>
```
