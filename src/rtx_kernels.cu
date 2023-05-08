#include <optix.h>
#include <math.h>
#include "rtx_params.h"


extern "C" static __constant__ Params params;

// min con closesthit
extern "C" __global__ void __raygen__rtx1() {
    //const uint3 idx = optixGetLaunchIndex();
    float &min = params.min;
    float &max = params.max;

    float3 ray_origin = make_float3(min, 0.0, 0.0);
    float3 ray_direction = make_float3(1.0, 0.0, 0.0);

    float tmin = 0;
    float tmax = max - min;
    float ray_time = 0;
    OptixVisibilityMask visibilityMask = 255;
    unsigned int rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
    unsigned int SBToffset = 0;
    unsigned int SBTstride = 0;
    unsigned int missSBTindex = 0;
    unsigned int payload = __float_as_uint(min);
    optixTrace(params.handle, ray_origin, ray_direction, tmin, tmax, ray_time,
            visibilityMask, rayFlags, SBToffset, SBTstride, missSBTindex, payload);

    *(params.output) = __uint_as_float(payload) + min;
}

extern "C" __global__ void  __closesthit__rtx() {
    float curr_tmax = optixGetRayTmax();
    optixSetPayload_0(__float_as_uint(curr_tmax));
}

extern "C" __global__ void  __miss__rtx() {
    optixSetPayload_0(__float_as_uint(INFINITY));
}




