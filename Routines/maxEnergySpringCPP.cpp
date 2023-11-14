#include <iostream>
#include <vector>
#include <algorithm> // for std::max_element

int maxEnergySpringCPP(int nodes, int maxncon, double *energy, int *damage, double damLimit) {
    double maxElement;
    int maxIndex;
    int rupture = 0;
    for (int i = 0; i < nodes*maxncon; ++i) {
        if (energy[i] > maxElement) {
            maxElement = energy[i];
            maxIndex = i;
        }
    }

    if(damLimit<maxElement){
        damage[maxIndex]=0;
        rupture = 10;
    }

    std::cout << "Max value: " << maxElement << " at position " << maxIndex << std::endl;

    return rupture;
}
