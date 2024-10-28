cmake_minimum_required(VERSION 3.28)
project(GPUroutines LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

#add_library
add_executable(GPUroutines 
    main.cpp
    GPUroutines.cu
    processKernel.cu
    GPUroutines.cuh
    processKernel.cuh
    )

target_compile_features(GPUroutines PUBLIC cxx_std_11)
set_target_properties(GPUroutines PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(GPUroutines PRIVATE CUDA::cufft)