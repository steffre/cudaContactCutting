#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"

__global__ void AddInts(double *p, int nodes, double sp, int nx, int ny, int nz){
    int ind1x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int ind2y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int ind3z = (blockIdx.z * blockDim.z) + threadIdx.z;
    int thn= ind1x+nx*ind2y+nx*ny*ind3z;

    if (ind1x < nx && ind2y<ny && ind3z<nz){
        p[thn*3 + 0] = ind1x*sp;
        p[thn*3 + 1] = ind2y*sp;
        p[thn*3 + 2] = ind3z*sp;
    }
}


int nodeGrid(int nodes, int nx, int ny, int nz, double *p, double sp){

    double *d_p;   /* p matrix that has the coordnate of the nodes as pointer*/

    //std::cout<<"\n nodes size : "<<nodes<<"\n";

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_p,sizeof(double) * nodes * 3);
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
    int threadsX = 16;                                   /*Thread per blocks on x dir */
    int threadsY = 16;                                   /*Thread per blocks on y dir */ 
    int threadsZ = 4;                                    /*Thread per blocks on z dir */ 
    int blocksX = nx/threadsX +1;                        /*Blocks on the x dir*/
    int blocksY = ny/threadsY +1;                        /*Blocks on the y dir*/
    int blocksZ = nz/threadsY +1;                        /*Blocks on the z dir*/

    dim3 THREADS( threadsX ,threadsY,threadsZ);
    dim3 BLOCKS(blocksX,blocksY,blocksZ);

    AddInts<<<THREADS, BLOCKS>>>(d_p,nodes,sp,nx,ny,nz);

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

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "AddInts launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaFree(d_p);

    return 0;
}
