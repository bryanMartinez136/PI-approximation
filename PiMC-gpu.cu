#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <curand.h>
#include <curand_kernel.h>


__global__ void init(unsigned int seed, curandState_t* states, int n) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
  curand_init(seed,
              i,
              0,
              &states[i]);
    }
}

// kernel to calculate random numbers
__global__ void randoms (curandState_t* states, float* numbers, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        numbers[i] = (curand_uniform(&states[i]));
    }

}
__global__ void calc_pi(float* rand_nums, float* sum, int n){

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        atomicAdd(sum, sqrt(1 - rand_nums[i]*rand_nums[i]));
    }
    
}

int main(int argc, char* argv[]){

    int N; 

// check for the appropriate number of command line arguments
    if (argc < 2){
        printf("Too few arguments\n"); 
        exit(1); 
    }

    N = atoi(argv[1]); 
    dim3 dimBlock(1024);
    dim3 dimGrid((int)ceil((float)N / 1024)); 

// check if the command line argument is a negative number. 
    if(N < 0){
        printf("No such thing as Negative Iterations !\n");
        exit(1);
    }

// keep track of seed value for every thread
  curandState_t* states;

  // allocate space on GPU for random states
  cudaMalloc((void**) &states, N*sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  init<<<dimGrid, dimBlock>>>(time(0), states, N);
  cudaDeviceSynchronize();

  // allocate array of unsigned ints on CPU and GPU
  float nums[N];
  float* dev_nums;
  cudaMalloc((void**) &dev_nums, N*sizeof(float));

  // obtain a uniformly random distriubtion of integers, maximum N
  randoms<<<dimGrid, dimBlock>>>(states, dev_nums, N);
  cudaDeviceSynchronize();

  // copy random distribution of integers back to host
  cudaMemcpy(nums, dev_nums, N*sizeof(float), cudaMemcpyDeviceToHost);
  

  // allocate for sum
  float sum = 0; 
  float* dev_sum; 
  cudaMalloc((void**) &dev_sum, sizeof(float)); 

  // copy the initial value to the gpu
  cudaMemcpy(dev_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

  // calling the kernel the first time
  calc_pi<<<dimGrid,dimBlock>>>(dev_nums, dev_sum, N);
  cudaDeviceSynchronize();

  // reset sum and give it back to device
  sum =0; 
  cudaMemcpy(dev_sum, &sum, sizeof(float), cudaMemcpyHostToDevice);

  // time to start timing
  cudaEvent_t start;
  cudaEventCreate(&start);
  cudaEvent_t stop;
  cudaEventCreate(&stop);

  // start timer
  cudaEventRecord(start,0);
  // call kernel second time to get the accurate timing
  calc_pi<<<dimGrid,dimBlock>>>(dev_nums, dev_sum, N);
  cudaDeviceSynchronize();

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);

  float diff;
  cudaEventElapsedTime(&diff, start, stop);
  printf("time: %f ms\n", diff);

  // deallocate timers
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  

  // copy minimum value to host
  cudaMemcpy(&sum, dev_sum, sizeof(float), cudaMemcpyDeviceToHost);

  float pi = sum/N *4 ; 
   printf("Pi: %f\n", pi); 

  cudaFree(states);
  cudaFree(dev_nums);
  cudaFree(dev_sum);

}
