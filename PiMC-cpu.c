#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/*
The PiMC programs will use one command line argument that corresponds to the
number of “iterations” used to compute pi with the Monte Carlo algorithm. The C
program must implement a serial code via a function call, and the CUDA program
must implement a parallel code via a kernel call. You must use a hierarchical
atomics strategy.

*/
float calc_pi(int N){
    float sum = 0; 
    float area; 

    for (int i = 0; i < N; i++) {

        float x = (float)rand() / RAND_MAX + rand() % 1;
        sum += sqrt(1- x*x);         

    }
    area = sum / N; 
    
    return area*4; 
}

int main(int argc, char* argv[]){
    int n; 
    time_t myTime, theEnd; 
    float pi; 
// check for the appropriate number of command line arguments
    if (argc < 2){
        printf("Too few arguments\n"); 
        exit(1); 
    }
    n = atoi(argv[1]); 
// check if the command line argument is a negative number. 
    if(n < 0){
        printf("No such thing as Negative Iterations !\n");
        exit(1);         
    }

    myTime = clock();

    pi = calc_pi(n); 
    printf("Pi : %f\n", pi); 

    theEnd = clock();  
    
    printf("Time elapsed = %f milliseconds \n", (float)(theEnd-myTime)/CLOCKS_PER_SEC);

    exit(0); 
}