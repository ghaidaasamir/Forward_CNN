#include "convolution.h"

__global__ void convolutionKernel(float *d_input, float *d_output, float *d_kernel, float *d_bias,
                                  int inputHeight, int inputWidth, int inputDepth,
                                  int kernelSize, int stride, int padding,
                                  int outputHeight, int outputWidth, int outputDepth)
{
    int outX = blockIdx.x * blockDim.x + threadIdx.x;
    int outY = blockIdx.y * blockDim.y + threadIdx.y;

    if (outX < outputWidth && outY < outputHeight)
    {

        for (int f = 0; f < outputDepth; ++f)
        {
            float sum = 0.0f;
            int startX = outX * stride - padding;
            int startY = outY * stride - padding;

            for (int c = 0; c < inputDepth; ++c)
            {
                for (int j = 0; j < kernelSize; ++j)
                {
                    for (int i = 0; i < kernelSize; ++i)
                    {
                        int x = startX + i;
                        int y = startY + j;

                        if (x >= 0 && x < inputWidth && y >= 0 && y < inputHeight)
                        {

                            int inputIndex = (c * inputHeight + y) * inputWidth + x;

                            int kernelIndex = ((f * inputDepth + c) * kernelSize + j) * kernelSize + i;
                            sum += d_input[inputIndex] * d_kernel[kernelIndex];
                        }
                    }
                }
            }

            sum += d_bias[f];
            int outputIndex = ((outY * outputWidth) + outX) * outputDepth + f;
            d_output[outputIndex] = sum;
        }
    }
}

void convolutionForwardCUDA(float *input, float *output, ConvolutionLayer convLayer)
{

    int inputHeight = convLayer.inputHeight;
    int inputWidth = convLayer.inputWidth;
    int inputDepth = convLayer.inputDepth;

    int outputHeight = convLayer.outputHeight;
    int outputWidth = convLayer.outputWidth;
    int outputDepth = convLayer.outputDepth;

    int kernelSize = convLayer.kernelSize;
    int stride = convLayer.stride;
    int padding = convLayer.padding;

    float *d_input, *d_output, *d_kernel, *d_bias;
    float *kernel = convLayer.kernel;
    
    cudaMalloc(&d_input, inputHeight * inputWidth * inputDepth * sizeof(float));
    cudaMalloc(&d_output, outputHeight * outputWidth * outputDepth * sizeof(float));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * inputDepth * outputDepth * sizeof(float));
    cudaMalloc(&d_bias, outputDepth * sizeof(float));

    cudaMemcpy(d_input, input, inputHeight * inputWidth * inputDepth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * inputDepth * outputDepth * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, convLayer.bias, outputDepth * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((outputWidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (outputHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    convolutionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_kernel, d_bias,
                                                          inputHeight, inputWidth, inputDepth,
                                                          kernelSize, stride, padding,
                                                          outputHeight, outputWidth, outputDepth);

    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, outputHeight * outputWidth * outputDepth * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_bias);
}
