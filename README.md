# RTX-CUDA-template
A simple RTX-CUDA interoperability test program based on the SAXPY example from NVIDIA's RTX Compute Samples https://github.com/NVIDIA/rtx_compute_samples

## Dependencies
- CUDA 11 or later
- OptiX 7.5 or later

## Compile and run
```
mkdir build && cd build
cmake ../ -DOPTIX_HOME=<PATH-TO-OPTIX-MAIN-DIR>
make
./rtx-cuda
```
