#include "fully_connected.h"
#include <algorithm>

__global__ void fullyConnectedKernel(float* input, float* output, float* weights, 
                                      float* bias, int inputSize, int outputSize){

     int idx = blockIdx.x * blockDim.x + threadIdx.x;
     if (idx < outputSize) {
          float sum = 0.0f;

          for (int i = 0; i < inputSize; ++i) {
               sum += input[i] * weights[i * outputSize + idx];
          }

          sum += bias[idx];
          output[idx] = sum;
    }
}

void fullyConnectedForwardCUDA(float* input, float* output, FullyConnectedLayer fcLayer){

     int inputSize = fcLayer.inputSize;
     int outputSize = fcLayer.outputSize;
     float* bias = fcLayer.bias; 
     float* weights = fcLayer.weights;

     int threadsPerBlock = 256;
     int blocksPerGrid = (outputSize + threadsPerBlock - 1) / threadsPerBlock;

     fullyConnectedKernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, weights, bias, inputSize, outputSize);

     cudaDeviceSynchronize(); 
     
}
