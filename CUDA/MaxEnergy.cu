#include <iostream>
#include <cuda_runtime.h>
#include "../Header/functions.h"

__global__ void maxValKernel(double *d_array, double *d_max, int *d_pos, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ double sdata[];

    // Load shared mem
    sdata[tid] = d_array[tid];
    __syncthreads();

    // Reduction in shared mem
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result for this block to global mem
    if (tid == 0) {
        *d_max = sdata[0];
        for (int i = 0; i < n; i++) {
            if (d_array[i] == sdata[0]) {
                *d_pos = i;
                break;
            }
        }
    }
}

int maxEnergySpring( int nodes,int maxncon, double *energy) {
    

    double *d_energy;
    cudaMalloc(&d_energy, nodes * maxncon * sizeof(double));
    cudaMemcpy(d_energy, energy, nodes * maxncon * sizeof(double), cudaMemcpyHostToDevice);

    double h_max;
    double *d_max;
    cudaMalloc(&d_max, sizeof(double));

    int h_pos;
    int *d_pos;
    cudaMalloc(&d_pos, sizeof(int));

    int threadsPerBlock = 512;
    int blocksPerGrid = (nodes * maxncon + threadsPerBlock - 1) / threadsPerBlock;
    maxValKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(double)>>>(d_energy, d_max, d_pos, nodes*maxncon);
    cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_pos, d_pos, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Max value: " << h_max << " at position " << h_pos << std::endl;

    cudaFree(d_energy);
    cudaFree(d_max);
    cudaFree(d_pos);

    

    return 0;
}