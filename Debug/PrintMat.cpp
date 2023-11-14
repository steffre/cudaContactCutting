 #include <iostream>

int printMat(double *p, int nx, int ny){

    for(int i=0; i<nx;i++){
        std::cout << "\n row: "<<i<<":";
        for( int j =0 ; j<ny;j++){
            std::cout<<"\t \t dir "<<j+1<<" : "<< p[i*3+j];
        }
    }

    return 0;
}