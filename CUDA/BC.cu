#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"

__global__ void BCdisp(double *p,int *TopNodes, int *BottomNodes, int nodes, double disp, double yh){
    
    int nodeidx = (blockIdx.x * blockDim.x) + threadIdx.x;

    int pIdx_y = nodeidx * 3 + 1;
    

    if (nodeidx < nodes){

        /*this is finding all the nodes in p that has the y position higher than yh - BoundSize */
        if (TopNodes[nodeidx] == 1){
            p[pIdx_y] = p[pIdx_y]+p[pIdx_y]*disp/yh;
        }
        if(BottomNodes[nodeidx] == 1){
            p[pIdx_y] = p[pIdx_y]+p[pIdx_y]*disp/yh;
        }
        
    }
}


int BC(int nodes, double *p,int *TopNodes, int *BottomNodes, double disp, double yh){

    double *d_p;            /* p matrix that has the coordnate of the nodes as pointer*/
    int *d_TopNodes;        /* TopNodes pointer to the memory that has top nodes saved*/
    int *d_BottomNodes;     /* TopNodes pointer to the memory that has top nodes saved*/
    

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_p,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_p! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_TopNodes,sizeof(int) * nodes);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_TopNodes! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_BottomNodes,sizeof(int) * nodes);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_BottomNodes! nope ... ";
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

    /* copy into the device memory */
    cudaStatus = cudaMemcpy(d_TopNodes,TopNodes, sizeof(int) * nodes, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy p into d_TopNodes into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_p);
        return 1;
    }

    /* copy into the device memory */
    cudaStatus = cudaMemcpy(d_BottomNodes,BottomNodes, sizeof(int) * nodes, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy p into d_BototmNodes into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_p);
        return 1;
    }

    /*louncing the kernel*/
    int threads = 1024;                                  /*Thread per blocks on x dir */
    int blocks = nodes/threads +1;                       /*Blocks on the x dir*/

    BCdisp<<<blocks, threads >>>(d_p,d_TopNodes,d_BottomNodes,nodes,disp,yh);

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

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "AddInts launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }


    cudaFree(d_p);
    cudaFree(d_TopNodes);
    cudaFree(d_BottomNodes);

    return 0;
}
