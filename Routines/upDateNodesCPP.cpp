#include <iostream>
#include <cmath>
#include <iomanip>
#include "../Header/functions.h"


int updateNodesCPP( int nodes, double *pnew, double *p, double *Fnodes,double deltat){
    
    for (int i=0;i < nodes*3; i++){
        pnew [i] =  p [ i ] + deltat*Fnodes[i];
        }
    return 0;
}