#include "pooling.h"

__global__ void maxPoolingKernel(float* input, float* output, int inputHeight, int inputWidth, int poolSize, int stride, int outputHeight, int outputWidth) {
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outX < outputWidth && outY < outputHeight) {
        int startX = outX * stride;
        int startY = outY * stride;

        float maxVal = -FLT_MAX;

        for (int i = 0; i < poolSize; ++i) {
            for (int j = 0; j < poolSize; ++j) {
                int x = startX + i;
                int y = startY + j;
                if (x < inputWidth && y < inputHeight) {
                    float element = input[y * inputWidth + x];
                    maxVal = max(maxVal, element);
                }
            }
        }

        output[outY * outputWidth + outX] = maxVal;
    }
}

void maxPoolingForwardCUDA(float* input, float* output, PoolingLayer poolingLayer) {

    int inputHeight = poolingLayer.inputHeight;
    int inputWidth = poolingLayer.inputWidth;
    int inputDepth = poolingLayer.inputDepth;
    
    int outputHeight = poolingLayer.outputHeight;
    int outputWidth = poolingLayer.outputWidth;
    int outputDepth = poolingLayer.outputDepth;
    
    int poolSize = poolingLayer.poolSize;  
    int stride = poolingLayer.stride;    

    dim3 threadsPerBlock(16, 16);  
    dim3 blocksPerGrid((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    maxPoolingKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, inputHeight, inputWidth, poolSize, stride, outputHeight, outputWidth);

    cudaDeviceSynchronize();
}
