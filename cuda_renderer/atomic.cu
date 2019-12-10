#include <stdio.h>
#define DSIZE 10
#define nTPB 256

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

typedef union  {
  float floats[2];                 // floats[0] = lowest
  int ints[4];                     // ints[1] = lowIdx
  unsigned long long int ulong;    // for atomic update
} my_atomics;

__device__ my_atomics test;


__device__ unsigned long long int my_atomicMin(unsigned long long int* address, float val1, int val2,int row)
{
	int idx = (blockDim.x * blockIdx.x) + threadIdx.x;    
	my_atomics loc, loctest;
  loc.floats[row] = val1;
  loc.ints[row+2] = val2;
  loctest.ulong = *address;
  while (loctest.floats[row] >  val1){
    loctest.ulong = atomicCAS(address, loctest.ulong,  loc.ulong);
  }
  return loctest.ulong;
}


__global__ void min_test(const float* data)
{

    int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
    if (idx < DSIZE)
      my_atomicMin(&(test.ulong), data[idx],idx,0);
    if (idx >= DSIZE && idx < 2*DSIZE)
      my_atomicMin(&(test.ulong), data[idx],idx,1);
}

int main() {

  float *d_data, *h_data;
  my_atomics my_init;
  my_init.floats[0] = 10.0f;
  my_init.ints[1] = DSIZE;

  h_data = (float *)malloc(2*DSIZE * sizeof(float));
  if (h_data == 0) {printf("malloc fail\n"); return 1;}
  cudaMalloc((void **)&d_data, DSIZE*2 * sizeof(float));
  cudaCheckErrors("cm1 fail");
  // create random floats between 0 and 1
  for (int i = 0; i < DSIZE; i++) h_data[i] = rand()/(float)RAND_MAX;
  cudaMemcpy(d_data, h_data, 2*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cmcp1 fail");
  cudaMemcpyToSymbol(test, &(my_init.ulong), sizeof(unsigned long long int));
  cudaCheckErrors("cmcp2 fail");
  min_test<<<(DSIZE+nTPB-1)/nTPB, nTPB>>>(d_data);
  cudaDeviceSynchronize();
  cudaCheckErrors("kernel fail");

  cudaMemcpyFromSymbol(&(my_init.ulong), test, sizeof(unsigned long long int));
  cudaCheckErrors("cmcp3 fail");

  printf("device min result = %f\n", my_init.floats[0]);
  printf("device idx result = %d\n", my_init.ints[2]);
  printf("device min result = %f\n", my_init.floats[1]);
  printf("device idx result = %d\n", my_init.ints[3]);
  float host_val = 10.0f;
  int host_idx = DSIZE;
  for (int i=0; i<DSIZE; i++)
    if (h_data[i] < host_val){
      host_val = h_data[i];
      host_idx = i;
      }

  printf("host min result = %f\n", host_val);
  printf("host idx result = %d\n", host_idx);
  return 0;
}
