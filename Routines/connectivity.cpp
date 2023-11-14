#include <iostream>
#include <cmath>
#include <omp.h>

int connectivity(int nodes, double ori, double *p, int *c){
    //omp_set_num_threads(8);

    int k=0;    
    double D,dx,dy,dz; /* distance of the nodes */

    for (int node1 = 0;node1 < nodes; node1++){
        //#pragma omp parallel for private(D);
        for (int node2 = node1+1; node2<nodes;node2++){
            dx = p[node1 * 3 + 0 ]- p[ node2 * 3 + 0 ];
            dy = p[node1 * 3 + 1 ]- p[ node2 * 3 + 1 ];
            dz = p[node1 * 3 + 2 ]- p[ node2 * 3 + 2 ];

            D = dx*dx + dy*dy + dz*dz;
            if (D <= ori*ori){
                #pragma omp critical
                {
                    c[ k + 0 ] = node1;
                    c[ k + 1 ] = node2;
                    k=k+2;
                }
                
            }
        }
    }

    return k/2;
}