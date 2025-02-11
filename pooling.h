#ifndef POOLING_H
#define POOLING_H

#include <cuda_runtime.h>
#include <cfloat>

struct PoolingLayer {
    int inputHeight;
    int inputWidth;
    int inputDepth;   
    int poolSize;      
    int stride;       
    int outputHeight;
    int outputWidth;
    int outputDepth;  
};

void maxPoolingForwardCUDA(float* input, float* output, PoolingLayer poolingLayer);

#endif
