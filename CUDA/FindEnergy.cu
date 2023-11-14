#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"
#include <cmath>

__global__ void findE(int nodes, int *Nncon, int *ncon, int maxncon, double *energy, double *l0, double *p,double mu, double alpha, int *damage){
    int nodeidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int Node1,Node2;
    double sb,DX,DY,DZ,l;
    
    if (nodeidx < nodes){

        //printf("Node: %d, Thread: %d, Nncon: %d\n", nodeidx, threadIdx.x, Nncon[nodeidx]);
        //printf("Node1: %d, Node2: %d\n", nodeidx*3, ncon[  nodeidx*maxncon + 0 ] * 3);
        
        for (int i=0;i<Nncon[nodeidx];i++){
            
            /*printf("Node: %d, i: %d, ncon: %d, energy: %f, l0: %f, p: %f, damage: %d\n",
               nodeidx, i, ncon[nodeidx*maxncon + i], energy[nodeidx*maxncon + i],
               l0[nodeidx*maxncon + i], p[nodeidx * 3], damage[nodeidx*maxncon + i]);
            */

            //printf("Node1: %d\n", nodeidx * 3);
            Node1 = nodeidx * 3;
            Node2 = ncon[  nodeidx*maxncon + i ] * 3;

            //printf("Node1: %d, Node2: %d\n", Node1, Node2);  // Print Node1 and Node2 indices


            DX = p[ Node1 + 0 ] - p[ Node2 + 0 ];
            DY = p[ Node1 + 1 ] - p[ Node2 + 1 ];
            DZ = p[ Node1 + 2 ] - p[ Node2 + 2 ];

            l = sqrt(DX*DX + DY*DY + DZ*DZ);

            //printf("l value for nodeidx %d: %f\n", nodeidx, l);

            sb = l / l0[  nodeidx*maxncon + i ];

            //N1 = alpha + 1;
            //N2 = - 0.5/alpha + 1; 

            
            if (sb > 1){
                energy[ nodeidx*maxncon + i ] = mu/alpha * ( pow(sb,alpha) + 2 *pow(1/sqrt(sb),alpha)) * damage[nodeidx * maxncon + i];
            }else{
                energy[ nodeidx*maxncon + i ] = 0;

            }
            
            //printf("Node: %d, DX: %f, DY: %f, DZ: %f, l: %f, sb: %f, N1: %f, N2: %f, En: %f\n", 
            //nodeidx, DX, DY, DZ, l, sb, N1, N2,energy[ nodeidx*maxncon + i ]);  // Added debug print

            
            //printf("Energy[%d]: %f\n", nodeidx*maxncon + i, energy[nodeidx*maxncon + i]);  // Print the calculated energy
        }
        //printf("Node: %d, Thread: %d, Nncon: %d - After Loop\n", nodeidx, threadIdx.x, Nncon[nodeidx]);

    }
}


int findEnergySpring(int nodes, int *Nncon,int *ncon, int maxncon, double *energy, double *l0, double *p,double mu, double alpha, int *damage){

    double *d_energy;   /* energy matrix that has the energy of the single spring */
    double *d_p;        /* p matrix that has the coordnate of the nodes as pointer */
    double *d_l0;        /* l0 matrix that has the lenght of the single spring */
    int *d_Nncon;       /* vector that tells for ech node how many nodes are connected */
    int *d_ncon;       /* vector that tells for ech node connection */
    int *d_damage;      /* vector that tells for ech spring which one is still active or not */

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_energy,sizeof(double) * nodes * maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_energy! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_p,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_p! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_l0,sizeof(double) * nodes * maxncon);
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

    cudaStatus = cudaMalloc(&d_ncon,sizeof(int) * nodes* maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_ncon! nope ... ";
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
    cudaStatus = cudaMemcpy(d_Nncon,Nncon, sizeof(int) * nodes, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy Nncon into d_Nncon into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_Nncon);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_ncon,ncon, sizeof(int) * nodes *maxncon, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy ncon into d_ncon into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_ncon);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_energy,energy, sizeof(double) * nodes *maxncon, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy energy into d_energy into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_energy);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_p,p, sizeof(double) * nodes * 3, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy p into d_p into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_p);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_l0,l0, sizeof(double) * nodes * maxncon, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy l0 into d_l0 into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_l0);
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

    findE<<< blocks, threads >>>(nodes,d_Nncon,d_ncon,maxncon,d_energy,d_l0,d_p,mu,alpha,d_damage);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "findE kernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_Nncon);
        cudaFree(d_energy);
        cudaFree(d_p);
        cudaFree(d_l0);
        cudaFree(d_ncon);
        cudaFree(d_damage);
        return 1;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching findE!\n";
        cudaFree(d_Nncon);
        cudaFree(d_energy);
        cudaFree(d_p);
        cudaFree(d_l0);
        cudaFree(d_ncon);
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

    cudaStatus = cudaMemcpy(energy, d_energy, sizeof(double) * nodes*maxncon, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        delete[] energy;
        cudaFree(d_Nncon);
        cudaFree(d_energy);
        cudaFree(d_p);
        cudaFree(d_l0);
        cudaFree(d_ncon);
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


    
    cudaFree(d_Nncon);
    cudaFree(d_energy);
    cudaFree(d_p);
    cudaFree(d_l0);
    cudaFree(d_ncon);
    cudaFree(d_damage);
    

    return 0;
}
