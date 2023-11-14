#include <iostream>
#include <cuda.h>
#include <ctime>
#include "../Header/functions.h"

__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__global__ void Fcompute(double *p, int *ncon, int *Nncon, double *Fnodes, double *l0,int nodes, int maxncon, double mu, double alpha, int *damage){
    
    int nodeidx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int Node1,Node2;
    double l,sb,fi,fix,fiy,fiz,DX,DY,DZ;
    

    if (nodeidx < nodes){
        /* initialization of the forces */
        Fnodes[ nodeidx * 3 + 0 ] = 0;
        Fnodes[ nodeidx * 3 + 1 ] = 0;
        Fnodes[ nodeidx * 3 + 2 ] = 0;
        
        for (int i=0; i < Nncon[ nodeidx ]; i++){
            Node1 = nodeidx * 3;
            Node2 = ncon[ nodeidx * maxncon + i ] * 3;

            DX = p[ Node1 + 0 ] - p[ Node2 + 0 ];
            DY = p[ Node1 + 1 ] - p[ Node2 + 1 ];
            DZ = p[ Node1 + 2 ] - p[ Node2 + 2 ];

            l = sqrt(DX*DX + DY*DY + DZ*DZ);

            //printf("l value for nodeidx %d: %f\n", nodeidx, l);

            sb = l / l0[ nodeidx * maxncon + i ];
            fi = (- mu * ( pow(sb,alpha) - pow(sb, -(1/2*alpha))))*damage[ nodeidx * maxncon + i ];

            fix = fi/l*DX;
            fiy = fi/l*DY;
            fiz = fi/l*DZ;

            atomicAdd_double(&Fnodes[nodeidx * 3 + 0], fix);
            atomicAdd_double(&Fnodes[nodeidx * 3 + 1], fiy);
            atomicAdd_double(&Fnodes[nodeidx * 3 + 2], fiz);

        }
        
        
    }
}


int assemblyForces(int nodes,int nc, double *Fnodes,int *ncon, int *Nncon, int maxncon, double *p,double mu, double alpha, double *l0, int *damage){

    double *d_p;        /* p matrix that has the coordnate of the nodes as pointer*/
    int *d_ncon;        /* nocn contain the nodes that are connected on the single node*/
    int *d_Nncon;       /* Nncon is a vector that contain the number of the nodes that are connected to aspecific node*/
    double *d_Fnodes;      /* Fnodes contains the force in x y z direction on each single node*/
    double *d_l0;       /* this contain the sigle spring lenght base on the ndoeal connection same size of ncon*/
    int *d_damage;       /* damage tanke in consideration the damaged springs*/
    

    

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc(&d_p,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_p! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_ncon,sizeof(int) * nodes * maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_ncon! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_damage,sizeof(int) * nodes * maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_damage! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_Nncon,sizeof(int) * nodes);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_Nncon! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_Fnodes,sizeof(double) * nodes * 3);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_Fnodes! nope ... ";
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    cudaStatus = cudaMalloc(&d_l0,sizeof(double) * nodes * maxncon);
    if(cudaStatus!= cudaSuccess){
        std::cout<<"\n Malloc d_l0! nope ... ";
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

    cudaStatus = cudaMemcpy(d_ncon,ncon, sizeof(int) * nodes * maxncon, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy ncon into d_ncon into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_ncon);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_damage,damage, sizeof(int) * nodes * maxncon, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy damage into d_damage into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_damage);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_Nncon,Nncon, sizeof(int) * nodes, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy Nncon into d_Nncon into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_Nncon);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_Fnodes,Fnodes, sizeof(double) * nodes * 3, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy Fnodes into d_Fnodes into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_Fnodes);
        return 1;
    }

    cudaStatus = cudaMemcpy(d_l0,l0, sizeof(double) * nodes * maxncon, cudaMemcpyHostToDevice );
    if(cudaStatus!= cudaSuccess){
        std::cout << "\n Could not copy l0 into d_l0 into the device !";
        std::cerr << "cudaMemcpy host to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(d_l0);
        return 1;
    }

    /* louncing the kernel */
    int threads = 512;                                   /* Thread per blocks on x dir */
    int blocks = (nodes+threads-1)/threads;              /* Blocks on the x dir*/

    Fcompute<<<blocks, threads >>>(d_p,d_ncon,d_Nncon,d_Fnodes,d_l0,nodes,maxncon,mu,alpha,d_damage);

    //gpuErrchk(cudaPeekAtLastError()); 
    //gpuErrchk(cudaDeviceSynchronize());

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching AddInts!\n";
        return 1;
    }

    cudaStatus = cudaMemcpy(Fnodes,d_Fnodes, sizeof(double) * nodes * 3, cudaMemcpyDeviceToHost) ;
    if(cudaStatus!= cudaSuccess){
        delete[] Fnodes;
        cudaFree(d_Fnodes);
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
    cudaFree(d_ncon);
    cudaFree(d_Nncon);
    cudaFree(d_Fnodes);
    cudaFree(d_l0);
    cudaFree(d_damage);

    return 0;
}
