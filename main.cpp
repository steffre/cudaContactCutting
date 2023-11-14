#include <iostream>
#include "Header/functions.h"
#include "Debug/debug.h"
#include <cmath>
#include <sys/stat.h> // Include for mkdir



/*  This is the main code to solve 3D structure with GPU computing 
    of a system of non linear spring that are attached to each other 
    the atachment is controlled by an are of influence */

int main(){

    // Saving data 
    std::string folderPath = "results/";  // Specify your folder here

    // Create the folder if it doesn't exist
    struct stat st = {0};
    if (stat(folderPath.c_str(), &st) == -1) {
        mkdir(folderPath.c_str(), 0700);  // 0700 are the permissions
    }

    
    /*---------------------------------*/
    /* initialization of the variables */
    /*---------------------------------*/
    
    /* declaration */
    double xl;        /*lenght of the speciment x direction (direction 0)*/
    double yh;        /*hight of the speciment in the y direction (direction 1)*/
    double zd;        /*depth of the speciment in the z direction (direction 2)*/
    double sp;        /*this parameter rappresent the spacing of the grid*/
    double ori;       /*this rappresent the orizon or the connectivity distance for each node*/
    double mu;        /*this parameter rappresent the hyperelastic shear modulus for the spring considering an Ogden behaviour*/
    double alpha;     /*this is the hardening parameter for the Ogden fromulation of the single spring*/
    double maxEnergy; /*max amount of energy before rupture of the bond */
    int NTopNodes;    /*N of nodes that are on the top boundary condition*/
    int NBottomNodes; /*N on nodes that are on the bottom boundary*/
    int oriN;         /*this rappresent unit of connection ex : oriN = 3 menas that ori = 3 times the initial spacing*/
    double BoundSize; /*sigificative lenght of the clamping area of the sample it should be sp * oriN*/
    int maxncon;      /*max number of connection per node*/
    double maxdisp;   /*this value rappresent the total diplacement targhet*/
    double disp;      /*this rappresent the actual displacement*/
    int itmax;        /*this rappresen the maximum number of iteraiton for the cycle*/
    double incdisp;   /*incremet of displacment for the steps*/
    int fun;          /*this will be used as a return value for some function to check that evrythign works*/
    int max_breaks;   /* max number of beam breaks for increment */

    /* initialization */
    
    oriN  = 3;
    xl    = 2;
    yh    = 1;
    zd    = 0.1;
    sp    = 0.05;
    mu    = 0.1;
    alpha = 2;
    maxEnergy = 0.6;
    maxdisp  = yh * 1.5;  /* this correspond to stretching the material to 200 % */ 
    itmax = 1000;
    disp  = 0;
    incdisp= maxdisp/itmax;
    max_breaks = 1000; 


    std::cout << " \n ---------------------------------------------------------------------- \n";
    std::cout << " -     Orizon : "<<oriN<<"                                                    - \n";
    std::cout << " -     Whide : "<<xl<<"                                        - \n";
    std::cout << " -     Height : "<<yh<<"                                                   - \n";
    std::cout << " -     Depth : "<<zd<<"                                                  - \n";
    std::cout << " -     Total Displacement : "<<maxdisp<<"                                                  - \n";
    std::cout << " -     Incremental Displacement : "<<incdisp<<"                                                  - \n";
    std::cout << " -     Number of steps : "<<itmax<<"                                                  - \n";
    std::cout << " ---------------------------------------------------------------------- \n";

    std::cout << " \n #### PARAMETER READ #### \n";

    /* ----- initialization of the dependent variable ----- */
    ori = sp * oriN;
    BoundSize = sp * oriN;
    maxncon = pow(((oriN)*2),3);
    std::cout<<"\n Max n connection : "<<ori<<"\n";
    std::cout<<"\n Max n connection : "<<maxncon<<"\n";
    int nodes = ( xl/sp + 1 ) *( yh/sp + 1 ) * ( zd/sp + 1 );  /*this is an overestimation of the exact number of nodes*/
    int nx = xl/sp + 1;   /*number of x grid or (direction 0)*/
    int ny = yh/sp + 1;   /*number of y grid or (direction 1)*/
    int nz = zd/sp + 1;   /*number of z grid or (direction 2)*/


    double *Ftot = new double [itmax*max_breaks];
    if (Ftot== nullptr) {
        std::cout << "\nFailed to allocate memory for TopNodes\n";
        return 1;
    }
    double *l0 = new double [nodes*maxncon];
    if (l0 == nullptr) {
        std::cout << "\nFailed to allocate memory for TopNodes\n";
        return 1;
    }
    int *ncon = new int [nodes*maxncon];
    if (ncon == nullptr) {
        std::cout << "\nFailed to allocate memory for ncon\n";
        return 1;
    }
    int *damage = new int [nodes*maxncon];
    if (damage == nullptr) {
        std::cout << "\nFailed to allocate memory for damage\n";
        return 1;
    }
    double *energy = new double [nodes*maxncon];
    if (energy == nullptr) {
        std::cout << "\nFailed to allocate memory for damage\n";
        return 1;
    }
    int *Nncon = new int [nodes];
    if (ncon == nullptr) {
        std::cout << "\nFailed to allocate memory for Nocn\n";
        return 1;
    }
    int *color = new int [nodes];
    if (color == nullptr) {
        std::cout << "\nFailed to allocate memory for color\n";
        return 1;
    }
    int *TopNodes    = new int[ nodes ];    /* top nodes as boundary condition as a pointer*/
    if (TopNodes == nullptr) {
        std::cout << "\nFailed to allocate memory for TopNodes\n";
        return 1;
    }
    std::fill(TopNodes, TopNodes + nodes, -1); /*initialization to -1 */
    int *BottomNodes = new int[ nodes ];    /* top nodes as boundary condition as a pointer*/
    if (BottomNodes == nullptr) {
        std::cout << "\nFailed to allocate memory for BottomNodes\n";
        return 1;
    }
    double *p = new double[nodes*3];   /* p matrix that has the coordnate of the nodes as pointer*/
    if (p == nullptr) {
        std::cout << "\nFailed to allocate memory for p\n";
        return 1;
    }

    int res = nodeGrid(nodes,nx,ny,nz,p,sp);
    if (res == 1){
        std::cout<<"\n it didn't work ! \n";
        return 0;
    }

    std::string baseFilename = "initial_p0";
    std::string currentFilename = baseFilename + ".dat";
    //writeMatrixToFile(currentFilename, p, nodes, 3);

    //int fun = printMat(p,nodes,3);

    std::cout << " \n #### GRID CREATED #### \n";
    /*calculation of the connectivity list*/
    int *c = new int[nodes*nodes*2];    /* connectivity matrix that has the connectivity list as pointer */
    
    int nc = connectivity( nodes, ori, p, c);

    baseFilename = "initial_c";
    currentFilename = baseFilename + ".dat";
    writeMatrixToFileInt(currentFilename, c, nc, 2);

    //fun = printMatInt(c,20,2);
    /*std::cout << " \n #### c list #### \n";
    for(int i=0;i<10;i++){
        std::cout << " \n ---- c ["<<i<<"]:"<<c[i]<<" -----";
    }*/

    //std::cout << " \n ---- NC :"<<nc<<" -----\n";
    std::cout << " \n #### CONNECTIVITY LIST CREATED #### \n";

    /*Calculaiton of the nodes that are on the bounday condiiton CUDA accelerated*/
    res=topB(nodes,ori,p,TopNodes,BoundSize,yh);
    if (res == 1){
        std::cout<<"\n topB it didn't work ! \n";
        return 0;
    }
    

    //baseFilename = "initial_TopNodes";
    //currentFilename = baseFilename + ".dat";
    //writeMatrixToFileInt(currentFilename, TopNodes, nodes, 1);
    //fun = printMatInt(TopNodes,nodes,1);

    std::cout << " \n #### TOP NODES CREATED #### \n";

    res=bottomB(nodes,ori,p,BottomNodes,BoundSize);
    if (res == 1){
        std::cout<<"\n topB it didn't work ! \n";
        return 0;
    }


    //baseFilename = "initial_BottomNodes";
    //currentFilename = baseFilename + ".dat";
    //writeMatrixToFileInt(currentFilename, BottomNodes, nodes, 1);
    //fun = printMatInt(BottomNodes,nodes,1);

    std::cout << " \n #### BOTTOM NODES CREATED #### \n";

    res = NodalConnection(nodes,nc,c, p,l0,Nncon,ncon,maxncon);

    std::cout << " \n #### DAMAGE #### \n";

    res = iniDamage(nodes, p, Nncon, ncon, maxncon, damage, yh, xl);

    std::cout << " \n #### ENERGY #### \n";

    res = iniEnergy(nodes, Nncon,  maxncon, energy);
    //baseFilename = "initial_Nncon";
    //currentFilename = baseFilename + ".dat";
    //writeMatrixToFileInt(currentFilename, Nncon, nodes, 1);

    //baseFilename = "initial_energy";
    //currentFilename = baseFilename + ".dat";
    //writeMatrixToFile(currentFilename, energy, nodes, maxncon);

    //baseFilename = "initial_l0";
    //currentFilename = baseFilename + ".dat";
    //writeMatrixToFile(currentFilename, l0, nodes, maxncon);


    /*for(int i=0;i<nodes;i++){
        for( int j=0;j<10;j++){
            std::cout << "ncon["<<i<<","<<j<<"] :"<<ncon[i*maxncon+j]<<"\t";
        }
    std::cout << " \n";          
    }
    std::cout << " \n \n #### Nncon #### \n \n";
    for(int i=0;i<nodes;i++){
        for( int j=0;j<8;j++){
            std::cout << "l0["<<i<<","<<j<<"] :"<<l0[i*maxncon+j]<<"\t";
        }
    std::cout << " \n";          
    }*/

    std::cout << " \n #### Nncon #### \n";
    //fun = printMatInt(Nncon,10,1);
    /*for(int i=0;i<nodes;i++){
        std::cout << " \n ---- Nncon ["<<i<<"]:"<<Nncon[i]<<" -----\n";
    }*/

    std::cout << " \n #### ncon #### \n";
    //fun = printMatInt(ncon,10,maxncon);
    /*for(int i=0;i<20;i++){
        std::cout << " \n ---- ncon :"<<ncon[i]<<" -----\n";
    }*/

    std::cout << " \n #### L0 #### \n";
    //fun = printMat(l0,10,8);
    
    std::cout << " \n #### NODAL CONNECTION AND L0 CREATED #### \n";

    std::cout << " \n ---------------------------------------------------------------------- \n";
    std::cout << " -     Nodes : "<<nodes<<"                                                   - \n";
    std::cout << " -     Nodal connection : "<<nc<<"                                       - \n";
    std::cout << " -     Orizon : "<<ori<<"                                                  - \n";
    std::cout << " -     Spacing : "<<sp<<"                                                 - \n";
    std::cout << " -     Boud Size : "<<BoundSize<<"                                               - \n";
    std::cout << " ---------------------------------------------------------------------- \n";
    

    std::cout << " \n #### INITIATION OF THE LOOP #### \n";
    
    int i = 0;  /* steps on the stretch */
    int j = 0;  /* steps on the broken beams */
    int k = 0;  /* total steps count */


    int rupture = 10;
    double percomplitoin = 0;
    baseFilename = "file_";
    std::string baseFilename2 = "fileColor_";
    //itmax=1;
    while (i<itmax && disp<maxdisp)
    {
        disp = disp + incdisp;
        /* setting the boundary condition */
        fun = BC(nodes,p,TopNodes,BottomNodes,incdisp,yh);

        /* loop untill all the bar are stable and no one is braking anymore */
        j=0;
        rupture = 10;
        
    
        while(j < max_breaks && rupture > 1.0){
            std::cout << " -     in the loop         - \n";
            res = solver(nodes,nc,c,p,mu, alpha,l0,ncon,Nncon,maxncon,TopNodes,BottomNodes, k, Ftot, damage);

            /* Calculation of the energy on the beams and brakage of the most loaded one */
            res = findEnergySpring(nodes,Nncon,ncon,maxncon,energy,l0,p,mu,alpha,damage);

            // baseFilename = "initial_Energy";
            // currentFilename = baseFilename + ".dat";
            // writeMatrixToFile(currentFilename, energy, nodes, maxncon);

            /* Calculation of the max enery and setting the damage to 0 on the max if larger than the threshhold */
            res = maxEnergySpringCPP( nodes,maxncon,energy,damage,maxEnergy);

            rupture = res;
            std::cout << " -     breaks : "<<j<<"            - \n";   

            j++;

            k++;

            res = colorDamage(nodes, color, damage,maxncon, Nncon);

            std::string currentFilename2 = folderPath + baseFilename2 + std::to_string(k) + ".dat";
            writeMatrixToFileInt(currentFilename2, color, nodes, 1);

            std::string currentFilename1 = folderPath + baseFilename + std::to_string(k) + ".dat";
            writeMatrixToFile(currentFilename1, p, nodes, 3);

            std::string baseFilename1 = folderPath + "totForce.dat";
        
            writeMatrixToFile(baseFilename1, Ftot, k, 1);
        }

        /* data for each step writing the results */

        /* printing on each loop */
        percomplitoin = disp/maxdisp;
        std::cout << " \n #### LOOP : " << i <<"complition percentage: "<< percomplitoin*100 <<" ####  iter :"<<res<<"\n";
        i++;
    }
    
    delete[] c;
    delete[] p;
    delete[] ncon;
    delete[] l0;
    delete[] Nncon;
    delete[] TopNodes;
    delete[] BottomNodes;

    return 0;

}