#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"
#include <cmath>

__global__ void nodCon(double *p,int nodes, double *l0, int *Nncon, int *ncon, int *c,int maxncon,int nc){
    int node = (blockIdx.x * blockDim.x) + threadIdx.x;
    int nodex = node * 3 + 0;
    int nodey = node * 3 + 1;
    int nodez = node * 3 + 2;
    
    int idxncon = node*maxncon;
    int k=0,node2,node2x,node2y,node2z;

    double xi,xj,yi,yj,zi,zj;
    
    if (node < nodes){
        /* in this soubroutine we need to pass by all the other nodes and calculate the l0 
        for the one that are conected and save the noide that is connected 
        and the how many nodes are connected */
        for (int j=0;j<nc;j++){
            
            if ( node == c[ j * 2 + 0 ] ){
                node2 = c[ j * 2 + 1 ];
                node2x = node2 * 3;
                node2y = node2 * 3 + 1;
                node2z = node2 * 3 + 2;

                xi = p[ nodex ];
                yi = p[ nodey ];
                zi = p[ nodez ];
                xj = p[ node2x ];
                yj = p[ node2y ];
                zj = p[ node2z ];

                l0 [ idxncon + k ] = pow( (xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj) , 0.5 );
                ncon [ idxncon + k ] = node2;
                k++;

                //printf("Thread %d: \t node: %d, \t l0[idxncon + k]: %f,\t  k: %d,\t j * 2: %d, \t c[j * 2 + 1]: %d\n", threadIdx.x, node, l0[idxncon + k-1],k,j*2, c[j * 2 + 1]);
                

            }
            if ( node == c [ j * 2 + 1 ] ){
                //printf("Thread %d: \t node: %d, \t c[j * 2]: %d,\t j * 2: %d, \t c[j * 2 + 1]: %d\n", threadIdx.x, node, c[j * 2],j*2, c[j * 2 + 1]);
                node2 = c [ j * 2 ];
                node2x = node2 * 3;
                node2y = node2 * 3 + 1;
                node2z = node2 * 3 + 2;

                xi = p[ nodex ];
                yi = p[ nodey ];
                zi = p[ nodez ];
                xj = p[ node2x ];
                yj = p[ node2y ];
                zj = p[ node2z ];

                l0 [ idxncon + k ] = pow( (xi-xj)*(xi-xj)+(yi-yj)*(yi-yj)+(zi-zj)*(zi-zj) , 0.5 );
                ncon [ idxncon + k ] = node2;
                k++;

                //printf("Thread %d: \t node: %d, \t l0[idxncon + k]: %f,\t  k: %d,\t j * 2: %d, \t c[j * 2]: %d\n", threadIdx.x, node, l0[idxncon + k-1],k,j*2, c[j * 2 ]);
                
            }
        }
        Nncon[node] = k;
        k=0;
    }
}


int NodalConnection(int nodes, int nc,  int *c, double *p, double *l0, int *Nncon, int *ncon, int maxncon){

    double *d_p;    /* p matrix that has the coordnate of the nodes as pointer */
    int *d_c;       /* connectivity matrix device pointer */
    double *d_l0;   /* l0 initial lenght of the springs */
    int *d_Nncon;   /* vector that tells for ech node how many nodes are connected */
    int *d_ncon;    /* vector that tells which nodes are connected */


    //std::cout<<"\n nodes size : "<<nodes<<"\n";
    //std::cout<<"\n Max n connection : "<<maxncon<<"\n";

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_p,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_p! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_c,sizeof(int) * nc * 2);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_c! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_l0,sizeof(double) * nodes*maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_l0! nope ... ";
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

    /* copy into the device memory */
    cudaStatus = cudaMemcpy(d_p,p, sizeof(double) * nodes * 3, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy p into d_p into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_p);
        return 1;
    }

    /* copy into the device memory */
    cudaStatus = cudaMemcpy(d_c,c, sizeof(int) * nc * 2, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy c into d_c into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_c);
        return 1;
    }

    cudaStatus = cudaMemset(d_l0, 0, sizeof(double) * nodes * maxncon);
    if(cudaStatus != cudaSuccess){
        std::cerr << "cudaMemset failed for d_l0: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMemset(d_Nncon, 0, sizeof(int) * nodes);
    if(cudaStatus != cudaSuccess){
        std::cerr << "cudaMemset failed for d_Nncon: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMemset(d_ncon, 0, sizeof(int) * nodes * maxncon);
    if(cudaStatus != cudaSuccess){
        std::cerr << "cudaMemset failed for d_ncon: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }


    /*louncing the kernel*/
    int threads = 512;                                   /*Thread per blocks on x dir */
    int blocks = (nodes+threads-1)/threads;                        /*Blocks on the x dir*/

    nodCon<<< blocks, threads >>>(d_p,nodes,d_l0,d_Nncon,d_ncon,d_c,maxncon,nc);

    //gpuErrchk(cudaPeekAtLastError()); 
    //gpuErrchk(cudaDeviceSynchronize());

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching nodCon!\n";
        return 1;
    }
    //std::cerr << "pre copy d_Nncon " << Nncon[0] << "\n";
    cudaStatus = cudaMemcpy(Nncon, d_Nncon, sizeof(int) * nodes, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        delete[] p;
        cudaFree(d_p);
        cudaFree(d_c);
        cudaFree(d_ncon);
        cudaFree(d_Nncon);
        cudaFree(d_l0);
        std::cerr << "Failed to copy d_Nncon from device to host: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }
    //std::cerr << "after copy d_Nncon " << Nncon[0] << "\n";

    cudaStatus = cudaMemcpy(ncon, d_ncon, sizeof(int) * nodes*maxncon, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        delete[] p;
        cudaFree(d_p);
        cudaFree(d_c);
        cudaFree(d_ncon);
        cudaFree(d_Nncon);
        cudaFree(d_l0);
        std::cerr << "Failed to copy d_ncon from device to host: " << cudaGetErrorString(cudaStatus) << "\n";
        return 1;
    }

    cudaStatus = cudaMemcpy(l0, d_l0, sizeof(double) * nodes*maxncon, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        delete[] p;
        cudaFree(d_c);
        cudaFree(d_p);
        cudaFree(d_ncon);
        cudaFree(d_Nncon);
        cudaFree(d_l0);
        std::cerr << "Failed to copy d_l0 from device to host: " << cudaGetErrorString(cudaStatus) << "\n";
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
    cudaFree(d_l0);
    cudaFree(d_c);

    return 0;
}
