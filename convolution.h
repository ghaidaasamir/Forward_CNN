#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cuda_runtime.h>
#include <vector>
#include <iostream>

struct ConvolutionLayer {
    int inputHeight;  // Height of the input feature map
    int inputWidth;   // Width of the input feature map
    int inputDepth;   // Depth of the input feature map (number of channels)
    int kernelSize;   // Size of the convolution kernel (e.g., 3x3)
    int stride;       // Stride for the convolution operation
    int padding;      // Padding applied to the input
    int outputHeight; // Height of the output feature map
    int outputWidth;  // Width of the output feature map
    int outputDepth;  // Number of output channels (filters)
    float* kernel;    // Pointer to the kernel weights (filters)
    float* bias;      // Pointer to the bias terms for each filter
};

void convolutionForwardCUDA(float* input, float* output, ConvolutionLayer convLayer);

#endif 
