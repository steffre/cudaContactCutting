#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"

__global__ void AddInts(double *p,int *BottomNodes, int nodes, double BoundSize){
    int nodeidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    int pIdx_y = nodeidx * 3 + 1;
    

    if (nodeidx < nodes){
        /*this is finding all the nodes in p that has the y position higher than yh - BoundSize */
        if (p[pIdx_y] <= (BoundSize)){
            BottomNodes[nodeidx] = 1;
        }
        else{
            BottomNodes[nodeidx] = 0;
        }
        
    }
}


int bottomB(int nodes,double ori, double *p,int *BottomNodes,double BoundSize){

    double *d_p;   /* p matrix that has the coordnate of the nodes as pointer*/
    int *d_BottomNodes; /* TopNodes pointer to the memory that has top nodes saved*/

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_p,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_p! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_BottomNodes,sizeof(int) * nodes);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_p! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    /* copy into the device memory */
    cudaStatus = cudaMemcpy(d_p,p, sizeof(double) * nodes * 3, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy p into d_p into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_p);
        return 1;
    }

    /*louncing the kernel*/
    int threads = 1024;                                   /*Thread per blocks on x dir */
    int blocks = nodes/threads +1;                       /*Blocks on the x dir*/

    AddInts<<<blocks, threads >>>(d_p,d_BottomNodes,nodes,BoundSize);

    //gpuErrchk(cudaPeekAtLastError()); 
    //gpuErrchk(cudaDeviceSynchronize());

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching AddInts!\n";
        return 1;
    }

    cudaStatus = cudaMemcpy(p,d_p, sizeof(double) * nodes * 3, cudaMemcpyDeviceToHost) ;
    if(cudaStatus!= cudaSuccess){
        delete[] p;
        cudaFree(d_p);
        std:: cout<<" \n Could not copy d_p back into the cpu !";
        std::cerr << "cudaMemcpy device to host failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    
    cudaStatus = cudaMemcpy(BottomNodes, d_BottomNodes, sizeof(int) * nodes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to copy d_TopNodes from device to host: " << cudaGetErrorString(cudaStatus) << "\n";
        cudaFree(d_p);
        cudaFree(d_BottomNodes);
        return 1;
    }

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "AddInts launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }


    cudaFree(d_p);
    cudaFree(d_BottomNodes);

    return 0;
}
