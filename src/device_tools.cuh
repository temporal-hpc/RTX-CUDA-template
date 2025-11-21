#pragma once
#include <iostream>
#define PRINT_LIMIT 32
__global__ void kernel_print_array_dev(int n, float *darray){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<n && i<PRINT_LIMIT; ++i){
        printf("tid %i --> array[%i] = %f\n", tid, i, darray[i]);
    }
    if(i < n){
        printf("...\n");
    }
}

__global__ void kernel_print_vertices_dev(int ntris, float3 *v){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i;
    if(tid != 0){
        return;
    }
    for(i=0; i<ntris && i<PRINT_LIMIT; ++i){
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+0, v[3*i+0].x, v[3*i+0].y, v[3*i+0].z);
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+1, v[3*i+1].x, v[3*i+1].y, v[3*i+1].z);
        printf("tid %i --> vertex[%i] = (%f, %f, %f)\n", tid, 3*i+2, v[3*i+2].x, v[3*i+2].y, v[3*i+2].z);
        printf("\n");
    }
    if(i < ntris){
        printf("...\n");
    }
}

void print_array_dev(int n, float *darray){
    printf("Printing random array:\n");
    kernel_print_array_dev<<<1,1>>>(n, darray);
    cudaDeviceSynchronize();
}

void print_vertices_dev(int ntris, float3 *devVertices){
    printf("Printing vertices:\n");
    kernel_print_vertices_dev<<<1,1>>>(ntris, devVertices);
    cudaDeviceSynchronize();
}

__global__ void kernel_gen_vertices(int ntris, float *array, float3 *vertices){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < ntris){
        int k = 3*tid;
        float xval = array[tid];
        vertices[k+0] = make_float3(xval,  0,  1);
        vertices[k+1] = make_float3(xval,  1, -1);
        vertices[k+2] = make_float3(xval, -1, -1);
    }
}

__global__ void kernel_gen_triangles(int ntris, float *array, uint3 *triangles){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < ntris){
        int k = 3*tid;
        triangles[tid] = make_uint3(k, k+1, k+2);
    }
}

float3* gen_vertices_dev(int ntris, float *darray){
    // vertices data
    float3 *devVertices = nullptr;
    auto alloc_status = cudaMalloc(&devVertices, sizeof(float3) * 3 * ntris);
    if (alloc_status != cudaSuccess) {
        std::cerr << __FILE__ << ":" << __LINE__
                  << " CUDA Error: '" << cudaGetErrorString(alloc_status)
                  << "' while allocating vertices (" << (3LL * ntris)
                  << " elements)\n";
        return nullptr;
    }

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_vertices<<<grid, block>>>(ntris, darray, devVertices);
    cudaDeviceSynchronize();
    return devVertices;
}


uint3* gen_triangles_dev(int ntris, float *darray){
    // data array
    uint3 *devTriangles = nullptr;
    auto alloc_status = cudaMalloc(&devTriangles, sizeof(uint3) * ntris);
    if (alloc_status != cudaSuccess) {
        std::cerr << __FILE__ << ":" << __LINE__
                  << " CUDA Error: '" << cudaGetErrorString(alloc_status)
                  << "' while allocating triangles (" << ntris
                  << " elements)\n";
        return nullptr;
    }

    // setup states
    dim3 block(BSIZE, 1, 1);
    dim3 grid((ntris+BSIZE-1)/BSIZE, 1, 1); 
    kernel_gen_triangles<<<grid, block>>>(ntris, darray, devTriangles);
    cudaDeviceSynchronize();
    return devTriangles;
}

float cpumin_vertex(int nv, float3 *dv){
    float3 *hv = new float3[nv];
    cudaMemcpy(hv, dv, sizeof(float3)*nv, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float min = 1000.0f;
    for(int i=0; i<nv; ++i){
        float val = hv[i].x;
        if(val < min){
            min = val;
        }
    }
    return min;
}


float cpumin_point(int np, float *dp){
    float *hp = new float[np];
    cudaMemcpy(hp, dp, sizeof(float)*np, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    float min = 1000000.0f;
    for(int i=0; i<np; ++i){
        //printf("Checking [%i] = %f\n", i, hp[i]);
        float val = hp[i];
        if(val < min){
            min = val;
            //printf("CPU new min %f\n", min);
        }
    }
    return min;
}

void cpuprint_array(int np, float *dp){
    float *hp = new float[np];
    cudaMemcpy(hp, dp, sizeof(float)*np, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0; i<np; ++i){
        printf("array [%i] = %f\n", i, hp[i]);
    }
}
