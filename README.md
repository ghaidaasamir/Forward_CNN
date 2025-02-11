# Forward_CNN

# CUDA Neural Network Layers Demo

This repository contains a basic demonstration of implementing three common neural network layers

**convolution**, **fully connected (dense)**, and **max pooling** using CUDA. 


### 1. **convolution.h & convolution.cu**

  Implements a convolution layer operation for a neural network.

### 2. **fully_connected.h & fully_connected.cu**

  Implements a fully connected (dense) layer for a neural network.

### 3. **pooling.h & pooling.cu**

  Implements a max pooling layer for a neural network.

### 4. **main.c**

  Simple test program that demonstrates how to use the above functions.


## Build and Run

**Build Directory and Configure:**
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

**Build the Project:**
   ```bash
   cmake --build .
   ```

**Run the Executable:**
   ```bash
   ./Forward_nn
   ```

## Summary

- **Convolution:** Applies filters to an input image/feature map.
- **Fully Connected:** Computes outputs from an input vector using dense connections.
- **Pooling:** Reduces the spatial size of the feature map by selecting maximum values.
