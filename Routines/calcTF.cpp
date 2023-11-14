#include <iostream>
#include <iostream>

#include <iomanip>
#include <omp.h>

double calculateF( int nodes, double *Fnodes, int *TopNodes){
    
    double Fytot=0;

    /*for (int i=0;i < nodes; i++){
        std::cout << " \n TopNodes ["<<i<<"]  = "<<TopNodes[i];
    }
    std::cout << " \n ------------\n\n";
    for (int i=0;i < nodes; i++){
        std::cout << " \n Fnodes ["<<i*3+1<<"]  = "<<Fnodes[i*3+1];
    }*/

    //#pragma omp parallel for
    for (int i=0;i < nodes; i++){
        if (TopNodes[i]==1){
            Fytot = Fytot + Fnodes [ i * 3 + 1 ];
            //std::cout << " \n Fy "<<Fytot;
        }
    }
    return Fytot;
}