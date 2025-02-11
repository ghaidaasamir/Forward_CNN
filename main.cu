#include <stdio.h>
#include <stdlib.h>
#include "convolution.h"
#include "fully_connected.h"
#include "pooling.h"

int main(void)
{
    int inputHeight = 4, inputWidth = 4, inputDepth = 1;
    int kernelSize = 3, stride = 1, padding = 0;
    int outputHeight = (inputHeight - kernelSize + 2 * padding) / stride + 1;
    int outputWidth = (inputWidth - kernelSize + 2 * padding) / stride + 1;
    int outputDepth = 1;

    float *h_input = (float *)malloc(inputHeight * inputWidth * inputDepth * sizeof(float));
    float *h_output = (float *)malloc(outputHeight * outputWidth * outputDepth * sizeof(float));
    float *h_kernel = (float *)malloc(kernelSize * kernelSize * inputDepth * outputDepth * sizeof(float));
    float *h_bias = (float *)malloc(outputDepth * sizeof(float));

    for (int i = 0; i < inputHeight * inputWidth * inputDepth; i++)
    {
        h_input[i] = 1.0f;
    }

    for (int i = 0; i < kernelSize * kernelSize * inputDepth * outputDepth; i++)
    {
        h_kernel[i] = 0.1111f;
    }

    for (int i = 0; i < outputDepth; i++)
    {
        h_bias[i] = 0.0f;
    }

    float *d_input, *d_output, *d_kernel, *d_bias;
    cudaMalloc((void **)&d_input, inputHeight * inputWidth * inputDepth * sizeof(float));
    cudaMalloc((void **)&d_output, outputHeight * outputWidth * outputDepth * sizeof(float));
    cudaMalloc((void **)&d_kernel, kernelSize * kernelSize * inputDepth * outputDepth * sizeof(float));
    cudaMalloc(&d_bias, outputDepth * sizeof(float));

    cudaMemcpy(d_input, h_input, inputHeight * inputWidth * inputDepth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * inputDepth * outputDepth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, outputDepth * sizeof(float), cudaMemcpyHostToDevice);

    ConvolutionLayer convLayer;
    convLayer.inputHeight = inputHeight;
    convLayer.inputWidth = inputWidth;
    convLayer.inputDepth = inputDepth;
    convLayer.kernelSize = kernelSize;
    convLayer.stride = stride;
    convLayer.padding = padding;
    convLayer.outputHeight = outputHeight;
    convLayer.outputWidth = outputWidth;
    convLayer.outputDepth = outputDepth;
    convLayer.kernel = d_kernel;
    convLayer.bias = d_bias;

    convolutionForwardCUDA(d_input, d_output, convLayer);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, outputHeight * outputWidth * outputDepth * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Convolution output (%dx%d):\n", outputHeight, outputWidth);
    for (int i = 0; i < outputHeight; i++)
    {
        for (int j = 0; j < outputWidth; j++)
        {
            printf("%f ", h_output[i * outputWidth + j]);
        }
        printf("\n");
    }

    int fc_inputSize = 4;
    int fc_outputSize = 2;

    float *h_fc_input = (float *)malloc(fc_inputSize * sizeof(float));
    float *h_fc_output = (float *)malloc(fc_outputSize * sizeof(float));
    float *h_fc_weights = (float *)malloc(fc_inputSize * fc_outputSize * sizeof(float));
    float *h_fc_bias = (float *)malloc(fc_outputSize * sizeof(float));

    for (int i = 0; i < fc_inputSize; i++)
    {
        h_fc_input[i] = 1.0f;
    }
    for (int i = 0; i < fc_inputSize * fc_outputSize; i++)
    {
        h_fc_weights[i] = 0.5f;
    }
    for (int i = 0; i < fc_outputSize; i++)
    {
        h_fc_bias[i] = 0.1f;
    }

    float *d_fc_input, *d_fc_output, *d_fc_weights, *d_fc_bias;
    cudaMalloc((void **)&d_fc_input, fc_inputSize * sizeof(float));
    cudaMalloc((void **)&d_fc_output, fc_outputSize * sizeof(float));
    cudaMalloc((void **)&d_fc_weights, fc_inputSize * fc_outputSize * sizeof(float));
    cudaMalloc(&d_fc_bias, fc_outputSize * sizeof(float));

    cudaMemcpy(d_fc_input, h_fc_input, fc_inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc_weights, h_fc_weights, fc_inputSize * fc_outputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc_bias, h_fc_bias, fc_outputSize * sizeof(float), cudaMemcpyHostToDevice);

    FullyConnectedLayer fcLayer;
    fcLayer.inputSize = fc_inputSize;
    fcLayer.outputSize = fc_outputSize;
    fcLayer.weights = d_fc_weights;
    fcLayer.bias = d_fc_bias;

    fullyConnectedForwardCUDA(d_fc_input, d_fc_output, fcLayer);
    cudaDeviceSynchronize();

    cudaMemcpy(h_fc_output, d_fc_output, fc_outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nFully Connected Layer output:\n");
    for (int i = 0; i < fc_outputSize; i++)
    {
        printf("%f ", h_fc_output[i]);
    }
    printf("\n");

    int poolSize = 2;
    int poolStride = 2;
    PoolingLayer poolingLayer;
    poolingLayer.inputHeight = outputHeight;
    poolingLayer.inputWidth = outputWidth;
    poolingLayer.inputDepth = outputDepth;
    poolingLayer.poolSize = poolSize;
    poolingLayer.stride = poolStride;
    poolingLayer.outputHeight = (outputHeight - poolSize) / poolStride + 1;
    poolingLayer.outputWidth = (outputWidth - poolSize) / poolStride + 1;
    poolingLayer.outputDepth = outputDepth;

    float *d_pool_output;
    float *h_pool_output = (float *)malloc(poolingLayer.outputHeight * poolingLayer.outputWidth * poolingLayer.outputDepth * sizeof(float));
    cudaMalloc((void **)&d_pool_output, poolingLayer.outputHeight * poolingLayer.outputWidth * poolingLayer.outputDepth * sizeof(float));

    maxPoolingForwardCUDA(d_output, d_pool_output, poolingLayer);
    cudaDeviceSynchronize();

    cudaMemcpy(h_pool_output, d_pool_output, poolingLayer.outputHeight * poolingLayer.outputWidth * poolingLayer.outputDepth * sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nMax Pooling output (%dx%d):\n", poolingLayer.outputHeight, poolingLayer.outputWidth);
    for (int i = 0; i < poolingLayer.outputHeight; i++)
    {
        for (int j = 0; j < poolingLayer.outputWidth; j++)
        {
            printf("%f ", h_pool_output[i * poolingLayer.outputWidth + j]);
        }
        printf("\n");
    }

    free(h_input);
    free(h_output);
    free(h_kernel);
    free(h_bias);

    free(h_fc_input);
    free(h_fc_output);
    free(h_fc_weights);
    free(h_fc_bias);

    free(h_pool_output);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_bias);

    cudaFree(d_fc_input);
    cudaFree(d_fc_output);
    cudaFree(d_fc_weights);
    cudaFree(d_fc_bias);

    cudaFree(d_pool_output);

    return 0;
}
