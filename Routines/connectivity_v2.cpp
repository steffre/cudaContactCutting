#include <iostream>
#include <cmath>
#include <omp.h>

int connectivity_v2(int nodes, double R, double *p, int *c){

    std::fill(c, c + nodes*nodes*2, -1); // Fill c with -1

    int num_threads = omp_get_num_procs();
    omp_set_num_threads(num_threads);

    int k=0;
    int** local_c = new int*[num_threads];
    int* local_k = new int[num_threads]();
    for(int i=0; i<num_threads; ++i) local_c[i] = new int[nodes*nodes];

    for (int node1=0; node1 < nodes; node1++){
        #pragma omp parallel for
        for (int node2 = node1+1; node2<nodes; node2++){
            double DX = ( p[ node1 * 3 + 0 ] - p[ node2 * 3 + 0 ])*( p[ node1 * 3 + 0 ] - p[ node2 * 3 + 0 ]);
            double DY = ( p[ node1 * 3 + 1 ] - p[ node2 * 3 + 1 ])*( p[ node1 * 3 + 1 ] - p[ node2 * 3 + 1 ]);
            double DZ = ( p[ node1 * 3 + 2 ] - p[ node2 * 3 + 2 ])*( p[ node1 * 3 + 2 ] - p[ node2 * 3 + 2 ]);
            double D = DX + DY + DZ;

            if (D<R*R){
                int id = omp_get_thread_num();
                #pragma omp critical
                if(local_k[id] < nodes*nodes){
                    local_c[id][local_k[id]*2] = node1;
                    local_c[id][local_k[id]*2 + 1] = node2;
                    local_k[id]++;
                }
                else{
                    std::cerr << "Error: local_k[" << id << "] index out of bounds!" << std::endl;
                    return -1;
                }

            }
        }
    }

    for(int i=0; i<num_threads; ++i){
        for(int j=0; j<local_k[i]; ++j){
            if(k < nodes*nodes*2){
                c[k] = local_c[i][j*2];
                k++;
                c[k] = local_c[i][j*2 + 1];
                k++;
            }
            else{
                std::cerr << "Error: k index out of bounds!" << std::endl;
                return -1;
            }
        }
    }

    if(k > nodes*nodes) {
        std::cerr << "Error: Connectivity list size exceeded number of nodes squared!" << std::endl;
        return -1;
    }

    for(int i=0; i<num_threads; ++i) delete[] local_c[i];  
    delete[] local_c;
    delete[] local_k;

    return k/2;
}
