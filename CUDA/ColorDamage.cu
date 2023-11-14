#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"
#include <cmath>

__global__ void col(int nodes, int maxncon, int *color,int *damage,int *Nncon){
    int nodeidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    if (nodeidx < nodes){

        color[nodeidx] = 0;
        
        for (int i=0;i<Nncon[nodeidx];i++){
            if (damage [nodeidx * maxncon + i] == 0){
                color[nodeidx] = 1;
            }
        }
    }
}


int colorDamage(int nodes, int *color, int *damage, int maxncon, int *Nncon){

    int *d_color;       /* vector that tells for ech node connection */
    int *d_Nncon;       /* vector that tells for ech node connection */
    int *d_damage;      /* vector that tells for ech spring which one is still active or not */

    cudaError_t cudaStatus;
    

    cudaStatus = cudaMalloc(&d_color,sizeof(int) * nodes);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_Nncon! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_Nncon,sizeof(int) * nodes);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_Nncon! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_damage,sizeof(int) * nodes * maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_Nncon! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    /* copy into the device memory */
    cudaStatus = cudaMemcpy(d_color,color, sizeof(int) * nodes, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy color into d_color into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_color);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_Nncon,Nncon, sizeof(int) * nodes, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy Nncon into d_Nncon into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_color);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_damage,damage, sizeof(int) * nodes * maxncon, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy damage into d_damage into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_damage);
        return 1;
    }

    

    /*louncing the kernel*/
    int threads = 512;                                   /*Thread per blocks on x dir */
    int blocks = (nodes + threads - 1) / threads;   // Ensure that all nodes are covered    

    col<<< blocks, threads >>>(nodes,maxncon,d_color,d_damage,d_Nncon);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "findE kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_color);
        cudaFree(d_Nncon);
        cudaFree(d_damage);
        return 1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching findE!\n";
        cudaFree(d_color);
        cudaFree(d_Nncon);
        cudaFree(d_damage);
        
        return 1;
    }
    
    //gpuErrchk(cudaPeekAtLastError()); 
    //gpuErrchk(cudaDeviceSynchronize());

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching nodCon!\n";
        return 1;
    }

    cudaStatus = cudaMemcpy(color, d_color, sizeof(int) * nodes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        delete[] color;
        cudaFree(d_color);
        cudaFree(d_Nncon);
        cudaFree(d_damage);
        
        std::cerr << "Failed to copy d_damage from device to host: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "nodCon launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }


    
    cudaFree(d_color);
    cudaFree(d_Nncon);
    cudaFree(d_damage);
    
    

    return 0;
}
