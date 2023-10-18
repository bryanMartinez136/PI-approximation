#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
/*
 1)
the filename that contains 10 million digits of pi and 2) the number of digits to be
evaluated. 
*/
void count_frequencies(FILE *fp, int n){

    int f_arr[10] = {0}; // put zero in array with indices 0-9
    char ch = fgetc(fp); 


    for(int i = 0; i < n; i++){
        f_arr[ch-48]++;
        ch = fgetc(fp);
    }

    for(int i = 0; i < 10; i++){
        printf("%d : %d\n", i, f_arr[i]); 
    }

}


int main(int argc, char*argv[]){

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

    count_frequencies(read_fp, n); 
    exit(0);

}
