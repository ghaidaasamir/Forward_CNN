cmake_minimum_required(VERSION 3.12)
project(CudaTest LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

set(SOURCES
    main.cu
    convolution.cu
    fully_connected.cu
    pooling.cu
)

add_executable(Forward_nn ${SOURCES})
