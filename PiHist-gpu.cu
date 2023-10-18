#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>



__global__ void frequencies(int* a, FILE* fp, int n){
    int i = 0; 
    if (i < 10) {
        a[i]++;  
    }

}

int main(int argc, char* argv[]){
    //----------------Rember to error handle--------------------
    if(argc < 3){
        printf("Not enough arguments\n");
        exit(1); 
    }
    
    // set the number of pi digits we want to count to this
    int n = atoi(argv[2]); 
    if(n < 0){
        printf("No such thing as a Negative Iterations !\n");
        exit(1);         
    }
    //open the file and check for error
    FILE * read_fp = fopen(argv[1], "r");
    if(read_fp == NULL){
        printf("The file does not exist.\n");
        exit(2);
    }


    // allocate space on cpu and gpu
    int nums[10] ={0};
    int* dev_nums;
    cudaMalloc((void**) &dev_nums, 10*sizeof(float));
    // create the block and grid
    dim3 dimBlock(1024);
    dim3 dimGrid((int)ceil((float)n / 1024)); 

    
    frequencies<<<dimGrid, dimBlock>>>(dev_nums, read_fp, n);
    cudaDeviceSynchronize();

    cudaMemcpy(nums, dev_nums, 10*sizeof(int), cudaMemcpyDeviceToHost);



    exit(0); 

}