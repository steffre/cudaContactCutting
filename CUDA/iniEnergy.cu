#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"
#include <cmath>

__global__ void inE(int nodes, int *Nncon, int maxncon, double *d_energy){
    int node = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idxncon = node*maxncon;
    
    if (node < nodes){
        
        for (int j=0;j<Nncon[node];j++){
            d_energy[idxncon + j ] = 0;
            
        }
    }
}


int iniEnergy(int nodes, int *Nncon, int maxncon, double *energy){

    double *d_energy;    /* p matrix that has the coordnate of the nodes as pointer */
    int *d_Nncon;   /* vector that tells for ech node how many nodes are connected */

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_energy,sizeof(double) * nodes * maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_energy! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_Nncon,sizeof(int) * nodes);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_Nncon! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    /* copy into the device memory */
    cudaStatus = cudaMemcpy(d_Nncon,Nncon, sizeof(int) * nodes, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy Nncon into d_Nncon into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_Nncon);
        return 1;
    }

    cudaStatus = cudaMemset(d_energy, 0, sizeof(double) * nodes * maxncon);
    if(cudaStatus != cudaSuccess){
        std::cerr << "cudaMemset failed for d_energy: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    /*louncing the kernel*/
    int threads = 512;                                   /*Thread per blocks on x dir */
    int blocks = (nodes+threads-1)/threads;                        /*Blocks on the x dir*/

    inE<<< blocks, threads >>>(nodes,d_Nncon,maxncon,d_energy);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "inD kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        // Handle error, free memory, etc.
        return 1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching inD!\n";
        // Handle error, free memory, etc.
        return 1;
    }
    
    //gpuErrchk(cudaPeekAtLastError()); 
    //gpuErrchk(cudaDeviceSynchronize());

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching nodCon!\n";
        return 1;
    }

    cudaStatus = cudaMemcpy(energy, d_energy, sizeof(int) * nodes*maxncon, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        delete[] energy;
        cudaFree(d_energy);
        cudaFree(d_Nncon);
        std::cerr << "Failed to copy d_damage from device to host: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "nodCon launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }


    
    cudaFree(d_Nncon);
    cudaFree(d_energy);

    return 0;
}
