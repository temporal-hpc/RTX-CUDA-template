#pragma once
#define SIM_MIN_X 0.0f
#define SIM_MAX_X 100.0f

__global__ void kernel_vertex_simulation(int ntris, float3 *v, curandState *state){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < ntris){
        // option a) random left and right (bounces on limit)
        //float dx = (curand_uniform(&state[tid])-0.5f)*SIM_MAX_X;

        // option b) just random movements in x+
        float dx = curand_uniform(&state[tid])*SIM_MAX_X;
        int k = 3*tid;
        float newx = v[k].x + dx;
        if(newx < SIM_MIN_X){
            newx -= 2.0f*dx; 
        }
        v[k+0] = make_float3(newx,  0,  1);
        v[k+1] = make_float3(newx,  1, -1);
        v[k+2] = make_float3(newx, -1, -1);
    }
}

__global__ void kernel_point_simulation(int n, float *p, curandState *state){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < n){
        // option a) random left and right (bounces on limit)
        //float dx = (curand_uniform(&state[tid])-0.5f)*SIM_MAX_X;

        // option b) just random movements in x+
        float dx = curand_uniform(&state[tid])*SIM_MAX_X;
        float val = p[tid] + dx;
        if(val < SIM_MIN_X){
            val -= 2.0*dx;
        }
        p[tid] = val;
    }
}
