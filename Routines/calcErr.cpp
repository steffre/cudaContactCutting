#include <iostream>
#include <cmath>
#include <iomanip>
#include <omp.h>

double calerr( int nodes, double *pnew, double *p){
    
    double err = 0;

    #pragma omp parallel for
    for (int i=0;i < nodes*3; i++){
        err = err + (p [ i ] - pnew[ i ] )*(p [ i ] - pnew[ i ] );
        }
    return sqrt(err);
}