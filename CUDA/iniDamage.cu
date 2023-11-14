#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"
#include <cmath>

__global__ void inD(double *p,int nodes, int *Nncon, int *ncon,int maxncon, int *d_damage, double yh, double xl){
    int node = (blockIdx.x * blockDim.x) + threadIdx.x;
    int idxncon = node*maxncon;
    
    if (node < nodes){
        //printf("Nncon[%d]: %d\n", node, Nncon[node]);
        for (int j=0;j<Nncon[node];j++){
            d_damage[idxncon + j ] = 1;
            if ( p[ node * 3 + 0 ]< xl/2 ){
                //printf("Node: %d, X: %f, Y: %f, xl/2: %f, yh/2: %f\n", node, p[node * 3 + 0], p[node * 3 + 1], xl/2, yh/2);
                if ((p[ node*3 + 1]< yh/2 && p[ ncon[ idxncon + j ] * 3 + 1]>= yh/2) ||
                    (p[ node*3 + 1]>= yh/2 && p[ ncon[ idxncon + j ] * 3 + 1]< yh/2)){
                    d_damage[idxncon + j ] = 0;  // Set damage to 1 if conditions are met
                }
            }
        }
    }
}


int iniDamage(int nodes, double *p, int *Nncon, int *ncon, int maxncon, int *damage, double yh, double xl){

    double *d_p;    /* p matrix that has the coordnate of the nodes as pointer */
    int *d_Nncon;   /* vector that tells for ech node how many nodes are connected */
    int *d_ncon;    /* vector that tells which nodes are connected */
    int *d_damage;    /* vector that tell what link to desable */

    std::cout << "yh: " << yh << " xl: " << xl << " nodes: " << nodes << std::endl;

    //std::cout<<"\n nodes size : "<<nodes<<"\n";
    //std::cout<<"\n Max n connection : "<<maxncon<<"\n";

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_p,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_p! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_Nncon,sizeof(int) * nodes);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_Nncon! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_ncon,sizeof(int) * nodes*maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_ncon! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_damage,sizeof(int) * nodes*maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_damage! nope ... ";
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

    cudaStatus = cudaMemcpy(d_Nncon,Nncon, sizeof(int) * nodes, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy Nncon into d_Nncon into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_Nncon);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_ncon,ncon, sizeof(int) * nodes * maxncon, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy ncon into d_ncon into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_ncon);
        return 1;
    }
    

    cudaStatus = cudaMemset(d_damage, 0, sizeof(int) * nodes * maxncon);
    if(cudaStatus != cudaSuccess){
        std::cerr << "cudaMemset failed for d_damage: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    /*louncing the kernel*/
    int threads = 512;                                   /*Thread per blocks on x dir */
    int blocks = (nodes+threads-1)/threads;                        /*Blocks on the x dir*/

    inD<<< blocks, threads >>>(d_p,nodes,d_Nncon,d_ncon,maxncon,d_damage,yh,xl);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "inD kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_p);
        cudaFree(d_ncon);
        cudaFree(d_Nncon);
        cudaFree(d_damage);
        return 1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching inD!\n";
        cudaFree(d_p);
        cudaFree(d_ncon);
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

    cudaStatus = cudaMemcpy(damage, d_damage, sizeof(int) * nodes*maxncon, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        delete[] p;
        cudaFree(d_p);
        cudaFree(d_ncon);
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


    cudaFree(d_p);
    cudaFree(d_ncon);
    cudaFree(d_Nncon);
    cudaFree(d_damage);

    return 0;
}
