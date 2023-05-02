#pragma once
template <typename IntegerType>
__device__ __host__ IntegerType roundUp(IntegerType x, IntegerType y) {
  return ((x + y - 1) / y) * y;
}

void launch_cuda(int n, float a, float *x, float *y);

std::string loadPtx(std::string filename) {
  std::ifstream ptx_in(filename);
  return std::string((std::istreambuf_iterator<char>(ptx_in)), std::istreambuf_iterator<char>());
}

struct GASstate {
  OptixDeviceContext context = 0;

  size_t temp_buffer_size = 0;
  CUdeviceptr d_temp_buffer = 0;
  CUdeviceptr d_temp_vertices = 0;
  //CUdeviceptr d_instances = 0;

  unsigned int triangle_flags = OPTIX_GEOMETRY_FLAG_NONE;

  OptixBuildInput triangle_input = {};
  OptixTraversableHandle gas_handle;
  CUdeviceptr d_gas_output_buffer;
  size_t gas_output_buffer_size = 0;

  OptixModule ptx_module = 0;
  OptixPipelineCompileOptions pipeline_compile_options = {};
  OptixPipeline pipeline = 0;

  OptixProgramGroup program_groups[3];
  OptixShaderBindingTable sbt = {};
};

void createOptixContext(GASstate &state) {
  CUDA_CHECK( cudaFree(0) ); // creates a CUDA context if there isn't already one
  OPTIX_CHECK (optixInit() ); // loads the optix library and populates the function table

  OptixDeviceContextOptions options = {};
  options.logCallbackFunction = &optixLogCallback;
  //options.logCallbackLevel = 4;
  options.logCallbackLevel = 1;

  OptixDeviceContext optix_context = nullptr;
  optixDeviceContextCreate(0, // use current CUDA context
                           &options, &optix_context);

  state.context = optix_context;
}

// load ptx and create module
void loadAppModule(GASstate &state) {
  std::string ptx = loadPtx(BUILD_DIR "/ptx/rtx_kernels.ptx");

  OptixModuleCompileOptions module_compile_options = {};
  module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
  module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  //module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MODERATE;
  module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  state.pipeline_compile_options.numPayloadValues = 1;
  state.pipeline_compile_options.numAttributeValues = 2; // 2 is the minimum
  state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  state.pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

  OPTIX_CHECK(optixModuleCreate(state.context, &module_compile_options,
                                       &state.pipeline_compile_options, ptx.c_str(),
                                       ptx.size(), nullptr, nullptr, &state.ptx_module));
}

void createGroupsClosestHit(GASstate &state) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  // raygen
  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = state.ptx_module;
  prog_group_desc[0].raygen.entryFunctionName = "__raygen__rtx1";

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[1].miss.module = state.ptx_module;
  prog_group_desc[1].miss.entryFunctionName = "__miss__rtx";


  // closest hit
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  prog_group_desc[2].hitgroup.moduleCH = state.ptx_module;
  prog_group_desc[2].hitgroup.entryFunctionNameCH = "__closesthit__rtx";
  prog_group_desc[2].hitgroup.moduleAH = nullptr;
  prog_group_desc[2].hitgroup.entryFunctionNameAH = nullptr;

  OPTIX_CHECK(optixProgramGroupCreate(state.context, prog_group_desc, 3, &program_group_options, nullptr, nullptr, state.program_groups));
}

// create Program groups
void createGroupsAnyHit(GASstate &state) {
  OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros
  OptixProgramGroupDesc prog_group_desc[3] = {};

  // raygen
  prog_group_desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  prog_group_desc[0].raygen.module = state.ptx_module;
  prog_group_desc[0].raygen.entryFunctionName = "__raygen__rtx2";

  // we need to create these but the entryFunctionNames are null
  prog_group_desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  prog_group_desc[1].miss.module = state.ptx_module;
  prog_group_desc[1].miss.entryFunctionName = "__miss__rtx";


  // closest hit
  prog_group_desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  prog_group_desc[2].hitgroup.moduleCH = nullptr;
  prog_group_desc[2].hitgroup.entryFunctionNameCH = nullptr;
  prog_group_desc[2].hitgroup.moduleAH = state.ptx_module;
  prog_group_desc[2].hitgroup.entryFunctionNameAH = "__anyhit__rtx";

  OPTIX_CHECK(optixProgramGroupCreate(state.context, prog_group_desc, 3, &program_group_options, nullptr, nullptr, state.program_groups));
}


void createPipeline(GASstate &state) {
  OptixPipelineLinkOptions pipeline_link_options = {};
  pipeline_link_options.maxTraceDepth = 1;
  OPTIX_CHECK(optixPipelineCreate(state.context, &state.pipeline_compile_options, &pipeline_link_options, state.program_groups, 3, nullptr, nullptr, &state.pipeline));
}

void populateSBT(GASstate &state) {
  char *device_records;
  CUDA_CHECK(cudaMalloc(&device_records, 3 * OPTIX_SBT_RECORD_HEADER_SIZE));

  char *raygen_record = device_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *miss_record = device_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE;
  char *hitgroup_record = device_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE;

  char sbt_records[3 * OPTIX_SBT_RECORD_HEADER_SIZE];
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[0], sbt_records + 0 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[1], sbt_records + 1 * OPTIX_SBT_RECORD_HEADER_SIZE));
  OPTIX_CHECK(optixSbtRecordPackHeader( state.program_groups[2], sbt_records + 2 * OPTIX_SBT_RECORD_HEADER_SIZE));

  CUDA_CHECK(cudaMemcpy(device_records, sbt_records, 3 * OPTIX_SBT_RECORD_HEADER_SIZE, cudaMemcpyHostToDevice));

  state.sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(raygen_record);

  state.sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(miss_record);
  state.sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  state.sbt.missRecordCount = 1;

  state.sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(hitgroup_record);
  state.sbt.hitgroupRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
  state.sbt.hitgroupRecordCount = 1;
}

void buildAccelerationStructure(GASstate &state, std::vector<float3> vertices, std::vector<uint3> triangles) {

  const size_t vertices_size = sizeof(float3) * vertices.size();
  CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&state.d_temp_vertices), vertices_size) );
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(state.d_temp_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice) );

  uint3* d_triangles;
  const size_t triangles_size = sizeof(uint3) * triangles.size();
  CUDA_CHECK( cudaMalloc(&d_triangles, triangles_size) );
  CUDA_CHECK( cudaMemcpy(d_triangles, triangles.data(), triangles_size, cudaMemcpyHostToDevice) );

  state.triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  state.triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  state.triangle_input.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
  state.triangle_input.triangleArray.vertexBuffers = &state.d_temp_vertices;
  state.triangle_input.triangleArray.flags = &state.triangle_flags;
  state.triangle_input.triangleArray.numSbtRecords = 1;
  state.triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  state.triangle_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(triangles.size());
  state.triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(d_triangles);

  OptixAccelBuildOptions accel_options = {};
  //accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK( optixAccelComputeMemoryUsage(state.context, &accel_options, &state.triangle_input, 1, &gas_buffer_sizes) );

  //void *d_temp_buffer_gas;
  state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;

  //CUDA_CHECK( cudaMalloc(&d_temp_buffer_gas, temp_size) );
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_temp_buffer), gas_buffer_sizes.tempSizeInBytes) );

  // non-compact output
  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8) );

  OptixAccelEmitDesc emitProperty = {};
  //emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitProperty.type = OPTIX_PROPERTY_TYPE_AABBS;
  emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

  OPTIX_CHECK( optixAccelBuild(
        state.context,
        0, 
        &accel_options, 
        &state.triangle_input,
        1,
        //state.d_temp_vertices,
        state.d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
	    d_buffer_temp_output_gas_and_compacted_size,
	    gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty, 1) 
  );  
 
  //CUDA_CHECK( cudaFree(d_temp_buffer_gas) );
  //size_t compacted_gas_size;
  //CUDA_CHECK( cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost) );

  /*
  if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
    CUDA_CHECK( cudaMalloc(&d_gas_output_buffer, compacted_gas_size) );

    // use handle as input and output
    OPTIX_CHECK( optixAccelCompact(optix_context, 0, gas_handle, reinterpret_cast<CUdeviceptr>(d_gas_output_buffer), compacted_gas_size, &gas_handle));

    CUDA_CHECK(cudaFree(d_buffer_temp_output_gas_and_compacted_size));
  } else {
    */
    state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    state.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
  //}
  //CUDA_CHECK(cudaFree(d_vertices));
}

void buildASFromDeviceData(GASstate &state, int nverts, int ntris, float3 *devVertices, uint3 *devTriangles) {

  //const size_t vertices_size = sizeof(float3) * vertices.size();
  //CUDA_CHECK( cudaMalloc(reinterpret_cast<void**>(&state.d_temp_vertices), vertices_size) );
  //CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(state.d_temp_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice) );
  state.d_temp_vertices = reinterpret_cast<CUdeviceptr>(devVertices);

  //uint3* d_triangles;
  //const size_t triangles_size = sizeof(uint3) * triangles.size();
  //CUDA_CHECK( cudaMalloc(&d_triangles, triangles_size) );
  //CUDA_CHECK( cudaMemcpy(d_triangles, triangles.data(), triangles_size, cudaMemcpyHostToDevice) );

  state.triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  state.triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  //state.triangle_input.triangleArray.numVertices = static_cast<unsigned int>(vertices.size());
  state.triangle_input.triangleArray.numVertices = static_cast<unsigned int>(nverts);
  state.triangle_input.triangleArray.vertexBuffers = &state.d_temp_vertices;
  state.triangle_input.triangleArray.flags = &state.triangle_flags;
  state.triangle_input.triangleArray.numSbtRecords = 1;
  state.triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  state.triangle_input.triangleArray.numIndexTriplets = static_cast<unsigned int>(ntris);
  state.triangle_input.triangleArray.indexBuffer = reinterpret_cast<CUdeviceptr>(devTriangles);

  OptixAccelBuildOptions accel_options = {};
  //accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

  OptixAccelBufferSizes gas_buffer_sizes;
  OPTIX_CHECK( optixAccelComputeMemoryUsage(state.context, &accel_options, &state.triangle_input, 1, &gas_buffer_sizes) );

  //void *d_temp_buffer_gas;
  state.temp_buffer_size = gas_buffer_sizes.tempSizeInBytes;

  //CUDA_CHECK( cudaMalloc(&d_temp_buffer_gas, temp_size) );
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&state.d_temp_buffer), gas_buffer_sizes.tempSizeInBytes) );

  // non-compact output
  CUdeviceptr d_buffer_temp_output_gas_and_compacted_size;
  size_t compactedSizeOffset = roundUp<size_t>(gas_buffer_sizes.outputSizeInBytes, 8ull);
  CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>(&d_buffer_temp_output_gas_and_compacted_size), compactedSizeOffset + 8) );

  OptixAccelEmitDesc emitProperty = {};
  //emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emitProperty.type = OPTIX_PROPERTY_TYPE_AABBS;
  emitProperty.result = (CUdeviceptr)((char *)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset);

  OPTIX_CHECK( optixAccelBuild(
        state.context,
        0, 
        &accel_options, 
        &state.triangle_input,
        1,
        state.d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
	    d_buffer_temp_output_gas_and_compacted_size,
	    gas_buffer_sizes.outputSizeInBytes,
        &state.gas_handle,
        &emitProperty, 1) 
  );  
 
  //CUDA_CHECK( cudaFree(d_temp_buffer_gas) );
  //size_t compacted_gas_size;
  //CUDA_CHECK( cudaMemcpy(&compacted_gas_size, (void *)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost) );

  /*
  if (compacted_gas_size < gas_buffer_sizes.outputSizeInBytes) {
    CUDA_CHECK( cudaMalloc(&d_gas_output_buffer, compacted_gas_size) );

    // use handle as input and output
    OPTIX_CHECK( optixAccelCompact(optix_context, 0, gas_handle, reinterpret_cast<CUdeviceptr>(d_gas_output_buffer), compacted_gas_size, &gas_handle));

    CUDA_CHECK(cudaFree(d_buffer_temp_output_gas_and_compacted_size));
  } else {
    */
    state.d_gas_output_buffer = d_buffer_temp_output_gas_and_compacted_size;
    state.gas_output_buffer_size = gas_buffer_sizes.outputSizeInBytes;
  //}
  //CUDA_CHECK(cudaFree(d_vertices));
}

void updateASFromHost(GASstate &state, std::vector<float3> vertices) {

  const size_t vertices_size = sizeof(float3) * vertices.size();
  CUDA_CHECK( cudaMemcpy(reinterpret_cast<void*>(state.d_temp_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice) );

  OptixAccelBuildOptions gas_accel_options = {};
  gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
  gas_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;

  OPTIX_CHECK( optixAccelBuild(
        state.context,
        0,
        &gas_accel_options,
        &state.triangle_input,
        1,
	    state.d_temp_buffer,
	    //state.d_temp_vertices,
        state.temp_buffer_size,
        state.d_gas_output_buffer,
        state.gas_output_buffer_size,
        &state.gas_handle,
        nullptr,
        0) );  
}


void updateASFromDevice(GASstate &state) {
    OptixAccelBuildOptions gas_accel_options = {};
    //gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    gas_accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    gas_accel_options.operation = OPTIX_BUILD_OPERATION_UPDATE;
    OPTIX_CHECK(optixAccelBuild(
        state.context,
        0,
        &gas_accel_options,
        &state.triangle_input,
        1,
        state.d_temp_buffer,
        //state.d_temp_vertices,
        state.temp_buffer_size,
        state.d_gas_output_buffer,
        state.gas_output_buffer_size,
        &state.gas_handle,
        nullptr,
        0)
    );  
}
