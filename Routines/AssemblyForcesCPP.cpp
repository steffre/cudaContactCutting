
#include <iostream>
#include <fstream>
#include "../Header/functions.h"


int assemblyForcesCPP(int nodes,int nc, double *Fnodes,int *ncon, int *Nncon, int maxncon, double *p,double mu, double alpha, double *l0){
 
    int node2;
    double l,sb,fi,fix,fiy,fiz,DX,DY,DZ;
    for( int node=0; node < nodes; node++){
        Fnodes[ node * 3 + 0 ] = 0.0;
        Fnodes[ node * 3 + 1 ] = 0.0;
        Fnodes[ node * 3 + 2 ] = 0.0;
    }
    std::ofstream outFile;
    outFile.open("output.txt");
    for( int node=0; node < nodes; node++){
        //Fnodes[ node * 3 + 0 ] = 0;
        //Fnodes[ node * 3 + 1 ] = 0;
        //Fnodes[ node * 3 + 2 ] = 0;

        for (int i=0; i < Nncon[ node ]; i++){
            //Node1 = node * 3;
            node2 = ncon[ node * maxncon + i ];

            DX = p[ node * 3 + 0 ] - p[ node2 * 3 + 0 ];
            DY = p[ node * 3 + 1 ] - p[ node2 * 3 + 1 ];
            DZ = p[ node * 3 + 2 ] - p[ node2 * 3 + 2 ];

            l = sqrt( DX*DX + DY*DY + DZ*DZ );

            //printf("l value for nodeidx %d: %f\n", nodeidx, l);

            sb = l / l0[ node * maxncon + i ];
            fi = - mu * ( pow(sb,alpha) - pow(sb, -(0.5*alpha)));

            fix = fi / l * DX;
            fiy = fi / l * DY;
            fiz = fi / l * DZ;

            outFile << "node1: " << node+1 << "\t node2: " << node2+1 << "\n";
            outFile << "DX: " << DX << " \t \t \t \t DY: " << DY << "\t \t \t \t DZ: " << DZ << "\n";
            outFile << "l: " << l << " \t \t \t \t sb: " << sb << " \t \t \t \t fi: " << fi << "\n";
            outFile << "fix: " << fix << " \t \t \t \t fiy: " << fiy << "\t \t \t \t fiz: " << fiz << "\n\n";


            Fnodes[ node * 3 + 0 ]=Fnodes[ node * 3 + 0 ] + fix;
            Fnodes[ node * 3 + 1 ]=Fnodes[ node * 3 + 1 ] + fiy;
            Fnodes[ node * 3 + 2 ]=Fnodes[ node * 3 + 2 ] + fiz;

            //Fnodes[ node2 * 3 + 0 ]=Fnodes[ node2 * 3 + 0 ] - fix;
            //Fnodes[ node2 * 3 + 1 ]=Fnodes[ node2 * 3 + 1 ] - fiy;
            //Fnodes[ node2 * 3 + 2 ]=Fnodes[ node2 * 3 + 2 ] - fiz;


        }

    }

        
    return 0;
        
}
