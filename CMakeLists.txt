cmake_minimum_required(VERSION 3.17)
project (CUDA-RTX-template LANGUAGES CXX CUDA)

# Set default build type to 'Release'
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# add path to our CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# turn on C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# for common headers
include_directories(./)
include_directories(./include)

# ==============================================================================
# Utility function to get PTX compilation & copying working
# ==============================================================================
include(PTXUtilities)

# ==============================================================================
# To link against CUDA::cudart and/or CUDA::cuda_driver
# ==============================================================================
find_package(CUDAToolkit REQUIRED)

# ==============================================================================
# Find Optix7
# ==============================================================================
find_package(OptiX7 REQUIRED)
include_directories(${OptiX7_INCLUDE_DIRS})


# ==============================================================================
# Find Thrust Library
# ==============================================================================
find_package(Thrust REQUIRED)


# ==============================================================================
# Find CUB Library
# ==============================================================================
find_package(CUB REQUIRED)

# ==============================================================================
# Main Executable
# ==============================================================================
thrust_create_target(Thrust)
add_executable(rtxcuda src/main.cu src/cuda_kernel.cu common/common.cpp)
target_link_libraries(rtxcuda PRIVATE CUDA::cudart OptiX7 Thrust)
set_target_properties(rtxcuda PROPERTIES CUDA_ARCHITECTURES native)

# ==============================================================================
# Add any needed PTX compilations here
# ==============================================================================
add_ptx_targets(rtxcuda rtx_kernels)
