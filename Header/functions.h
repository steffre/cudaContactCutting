#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#include <iostream>
#include <cmath>

int nodeGrid(int nodes, int nx, int ny, int nz, double *p, double sp);
int topB( int nodes,double ori,double *p,int *TopNodes,double BoundSize,double yh);
int bottomB( int nodes,double ori,double *p,int *BottomNodes,double BoundSize);
int NodalConnection(int nodes, int nc,  int *c, double *p, double *l0, int *Nncon, int *ncon, int maxncon);
int BC(int nodes, double *p,int *TopNodes, int *BottomNodes, double disp,double yh);
int assemblyForces(int nodes,int nc, double *Fnodes,int *ncon, int *Nncon, int maxncon, double *p,double mu, double alpha, double *l0, int *damage);
int applyBC(double *Fnodes, int *TopNodes, int *BottomNodes, int nodes);
int updateNodes(double *Fnodes,double *p,double *pnew, double deltat, int nodes);
int updateP(double *p,double *pnew, int nodes);
int iniDamage(int nodes, double *p, int *Nncon, int *ncon, int maxncon, int *damage, double yh, double xh);
int iniEnergy(int nodes, int *Nncon, int maxncon, double *energy);
int colorDamage(int nodes, int *color, int *damage, int maxncon, int *Nncon);

int connectivity_v2(int nodes, double R, double *p, int *c);
int connectivity(int nodes, double ori, double *p, int *c);
double calerr( int nodes, double *pnew, double *p);
int solver(int nodes, int nc, int *c, double *p, double mu, double alpha, double *l0, int *ncon, int *Nncon, int maxncon, int *TopNodes, int *BottomNodes, int step, double *Ftot, int *damage);
double calculateF( int nodes, double *Fnodes, int *TopNodes);
void writeMatrixToFile(const std::string& filename, double *matrix, int nodes, int col);
void writeMatrixToFileInt(const std::string& filename, int *matrix, int nodes, int colum);
int assemblyForcesCPP(int nodes,int nc, double *Fnodes,int *ncon, int *Nncon, int maxncon, double *p,double mu, double alpha, double *l0);
int updateNodesCPP( int nodes, double *pnew, double *p, double *Fnodes,double deltat);
int findEnergySpring(int nodes, int *Nncon,int *ncon, int maxncon, double *energy, double *l0, double *p,double mu, double alpha, int *damage);
int maxEnergySpring( int nodes,int maxncon, double *energy);
int maxEnergySpringCPP(int nodes, int maxncon, double *energy, int *damage, double damLimit);
 


#endif