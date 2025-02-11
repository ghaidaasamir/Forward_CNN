#ifndef FULLY_CONNECTED_H
#define FULLY_CONNECTED_H

#include <cuda_runtime.h>

struct FullyConnectedLayer {
    int inputSize;      
    int outputSize;     
    float* weights;     
    float* bias;   
};

__global__ void fullyConnectedKernel(float* input, float* output, float* weights, 
                                      float* bias, int inputSize, int outputSize);

void fullyConnectedForwardCUDA(float* input, float* output, FullyConnectedLayer fcLayer);

#endif 
