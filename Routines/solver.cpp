#include <iostream>
#include "../Header/functions.h"

int solver(int nodes, int nc, int *c, double *p, double mu, double alpha, double *l0, int *ncon, int *Nncon, int maxncon, int *TopNodes, int *BottomNodes, int step, double *Ftot, int *damage){

    /* initializatoin of the paramenter for the iterative solution */
    double deltat= 0.001;                                           /* delta step on the adjustment of the nodes */
    double toll = 1e-5;                                             /* tollerance for the convergency of the program */
    double err = toll*10;                                           /* error in the position of the nodes */
    int iteration = 0;                                              /* iteration counter */
    int itmax = 10000;                                              /* max iteration allowed in the loop */
    int res;


    double *Fnodes = new double [ nodes * 3];      /* This contain the force in direction 1 X , direction 2 Y and direction 3Z*/
    double *pnew = new double [nodes * 3];         /* This vector contain the nodal position in direction 1 X, direction 2 Y and direction 3 Z*/
    double *pcheck = new double [nodes * 3];       /* This vector contain the nodal position in direction 1 X, direction 2 Y and direction 3 Z*/
    

    std::string baseFilename = "preIterAfterBC_";
    std::string currentFilename = baseFilename + std::to_string(1) + ".dat";
        
    writeMatrixToFile(currentFilename, p, nodes, 3);
    //std::cout << " \n #### loop start  #### \n";
    while( err > toll & iteration< itmax){
        
        res = assemblyForces(nodes,nc, Fnodes,ncon,Nncon,maxncon,p,mu,alpha,l0,damage);    /* function that assemble all the forces ona a single node */
             
        //res = assemblyForcesCPP( nodes,nc,Fnodes,ncon, Nncon,maxncon,p,mu,alpha,l0);

        res = applyBC(Fnodes,TopNodes,BottomNodes,nodes);                           /* boundary condition application */

        baseFilename = "forcesSeq_";
        currentFilename = baseFilename + std::to_string(iteration) + ".dat";
        //writeMatrixToFile(currentFilename, Fnodes, nodes, 3);

        res = updateNodes(Fnodes,p,pnew,deltat,nodes);                              /* updating the nodes position */
        
        baseFilename = "updatepCuda_";
        currentFilename = baseFilename + std::to_string(iteration) + ".dat";
        //writeMatrixToFile(currentFilename, pnew, nodes, 3);
        
        //res = updateNodesCPP(nodes,pnew,p,Fnodes,deltat); 

        //baseFilename = "updatepCPP_";
        //currentFilename = baseFilename + std::to_string(iteration) + ".dat";
        //writeMatrixToFile(currentFilename, pnew, nodes, 3);
        
        err = calerr(nodes,p,pnew);                                                 /* calculationg the error */
        std::cout << " \n it :"<<iteration<<"#### err done #### "<<err<<"\n";
        
        res = updateP(p,pnew,nodes);                                                /* updating the pold to p new */
    
        

        iteration++;
        
        
    }



    res = assemblyForces(nodes,nc, Fnodes,ncon,Nncon,maxncon,p,mu,alpha,l0,damage);
    /* calculate the final force acting on the top */
    Ftot[step] = calculateF(nodes,Fnodes,TopNodes);

    /* calculate the energy on the spring */

    //baseFilename = "finalp_";
    //currentFilename = baseFilename + std::to_string(iteration) + ".dat";    
    //writeMatrixToFile(currentFilename, p, nodes, 3);
    
    //baseFilename = "finalforce_";
    //currentFilename = baseFilename + std::to_string(iteration) + ".dat";
    //writeMatrixToFile(currentFilename, Fnodes, nodes, 3);

    std::cout << " \n #### Loop done it  #### "<<iteration<<" \t Total Force = "<<Ftot[step]<<" ----- err : "<<err<<" \n";

    delete[] Fnodes;
    delete[] pnew;

    return iteration;

}