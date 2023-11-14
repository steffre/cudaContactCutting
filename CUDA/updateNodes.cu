#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"

__global__ void updateNodeskernel(double *Fnodes,double *p, double *pnew, int nodes, double deltat){
    
    int nodeidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (nodeidx < nodes * 3){

        pnew[nodeidx] = p[nodeidx]+Fnodes[nodeidx]*deltat;
        
    }
}


int updateNodes(double *Fnodes,double *p,double *pnew, double deltat, int nodes){

    double *d_Fnodes;       /* Vector of the forces */
    double *d_p;            /* Vector of the position of the nodes */
    double *d_pnew;         /* New posiiton after apply force */
    

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_Fnodes,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_Fnodes! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_p,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_p! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_pnew,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_pnew! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    /* copy into the device memory */
    cudaStatus = cudaMemcpy(d_Fnodes,Fnodes, sizeof(double) * nodes * 3, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy Fnodes into d_Fnodes into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_Fnodes);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_p,p, sizeof(double) * nodes * 3, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy p into d_TopNodes into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_p);
        return 1;
    }

   

    /*louncing the kernel*/
    int threads = 1024;                                   /* Thread per blocks on x dir */
    int blocks = (nodes*3)/threads+1;              /* Blocks on the x dir*/

    //std::cout << "Nodes: " << nodes << std::endl;
    //std::cout << "Threads: " << threads << std::endl;
    //std::cout << "Blocks: " << blocks << std::endl;


    updateNodeskernel<<< blocks, threads >>>(d_Fnodes,d_p,d_pnew,nodes,deltat);

    //gpuErrchk(cudaPeekAtLastError()); 
    //gpuErrchk(cudaDeviceSynchronize());

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching AddInts!\n";
        return 1;
    }

    cudaStatus = cudaMemcpy(pnew,d_pnew, sizeof(double) * nodes * 3, cudaMemcpyDeviceToHost) ;
    if(cudaStatus!= cudaSuccess){
        delete[] pnew;
        cudaFree(d_pnew);
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


    cudaFree(d_Fnodes);
    cudaFree(d_p);
    cudaFree(d_pnew);

    return 0;
}
