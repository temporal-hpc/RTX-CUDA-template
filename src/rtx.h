#pragma once
void rtx(int n, int k, int steps, int alg, float *darray, curandState *devStates) {
    printf("--------------------- RTX OptiX Min %-15s ---------------------\n", algStr[alg]);
    Timer timer;
    float output, *d_output, cpuMin=-1.0f;

    // 1) Generate geometry from device data
    printf("Generating geometry.................."); fflush(stdout);
    timer.restart();
    float3* devVertices = gen_vertices_dev(n, darray);
    uint3 *devTriangles = gen_triangles_dev(n, darray);
    if (devVertices == nullptr || devTriangles == nullptr) {
        printf("failed\n");
        printf("Insufficient device memory for OptiX geometry (requested %lld vertices, %d triangles).\n",
               3LL * n, n);
        return;
    }
    //print_array_dev(n, p.first);
    //print_vertices_dev(n, devVertices);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());


    // 2) RTX OptiX Config (ONCE)
    printf("RTX Config..........................."); fflush(stdout);
    timer.restart();
    GASstate state;
    createOptixContext(state);
    loadAppModule(state);
	createGroupsClosestHit(state);
    createPipeline(state);
    populateSBT(state);
    timer.stop();
    printf("done: %f ms\n",timer.get_elapsed_ms());




    // 3) Build Acceleration Structure 
    printf("%sBuild AS on GPU......................", AC_MAGENTA); fflush(stdout);
    timer.restart();
    buildASFromDeviceData(state, 3*n, n, devVertices, devTriangles);
    cudaDeviceSynchronize();
    timer.stop();
    printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);


    // 4) Populate and move parameters to device (ONCE)
    printf("device params to GPU "); fflush(stdout);
    CUDA_CHECK( cudaMalloc(&d_output, sizeof(float)) );
    timer.restart();
    Params params;
    params.handle = state.gas_handle;
    params.min = 0;
    params.max = 100000000;
    params.output = d_output;
    params.k = k;
    Params *device_params;
    printf("(%7.3f MB)....", (double)sizeof(Params)/1e3); fflush(stdout);
    CUDA_CHECK(cudaMalloc(&device_params, sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(device_params, &params, sizeof(Params), cudaMemcpyHostToDevice));
    timer.stop();
    printf("done: %f ms\n", timer.get_elapsed_ms());


    // 5) Simulation
    printf("Simulating for %i steps\n", steps);
    for(int ki = 0; ki<steps; ++ki){
        //print_vertices_dev(n, (float3*)state.d_temp_vertices);
        // 5.1) OptiX launch `W` waves of rays
        cpuMin = cpumin_vertex(3*n, devVertices);
        for(int ki = 0; ki<RTX_REPEATS; ++ki){
            printf("\t%sOptiX Launch [%-15s].......", AC_BOLDCYAN, algStr[alg]); fflush(stdout);
            timer.restart();
            OPTIX_CHECK(optixLaunch(state.pipeline, 0, reinterpret_cast<CUdeviceptr>(device_params), sizeof(Params), &state.sbt, 1, 1, 1));
            CUDA_CHECK(cudaDeviceSynchronize());
            timer.stop();
            CUDA_CHECK( cudaMemcpy(&output, d_output, sizeof(float), cudaMemcpyDeviceToHost) );
            printf("done: %f ms (min %f, cpuMin %f)%s\n", timer.get_elapsed_ms(), output, cpuMin, AC_RESET);
        }
        //printf("Press enter...\n");
        //getchar();

        // 5.2) Simulation --> update geometry with CUDA kernel 
        printf("\tParticles Random Movement............"); fflush(stdout);
        dim3 block(BSIZE, 1, 1);
        dim3 grid((n + BSIZE-1)/BSIZE, 1, 1);
        timer.restart();
        kernel_vertex_simulation<<<grid, block>>>(n, (float3*)state.d_temp_vertices, devStates);
        cudaDeviceSynchronize();
        timer.stop();
        printf("done: %f ms\n", timer.get_elapsed_ms());
        //printf("Press enter...\n");
        //getchar();

        // 5.3) update AS from device data
        printf("\t%sUpdating AS..........................", AC_YELLOW); fflush(stdout);
        timer.restart();
        updateASFromDevice(state);
        CUDA_CHECK(cudaDeviceSynchronize());
        timer.stop();
        printf("done: %f ms%s\n", timer.get_elapsed_ms(), AC_RESET);
        printf("\n");
        //printf("Press enter...\n");
        //getchar();
    }
    printf("done\n");
    // 6) clean up
    printf("cleaning up RTX environment.........."); fflush(stdout);
    OPTIX_CHECK(optixPipelineDestroy(state.pipeline));
    for (int i = 0; i < 3; ++i) {
        OPTIX_CHECK(optixProgramGroupDestroy(state.program_groups[i]));
    }
    OPTIX_CHECK(optixModuleDestroy(state.ptx_module));
    OPTIX_CHECK(optixDeviceContextDestroy(state.context));

    CUDA_CHECK(cudaFree(device_params));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(state.sbt.raygenRecord)));
    printf("done: %f ms\n", timer.get_elapsed_ms());
    printf("-------------------------------------------------------------------------\n\n");
}
