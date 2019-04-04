// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void global_reduce_kernel_v0(float * d_out, float * d_in)
{
  //TODO Implementar version 0 reduction, esquema imagen 1, utilizando memoria global
}

__global__ void global_reduce_kernel_v1(float * d_out, float * d_in)
{
  //TODO Implementar version 1 reduction, esquema imagen 2, utilizando memoria global
}

__global__ void shmem_reduce_kernel(float * d_out, const float * d_in)
{
  //TODO Implementar reduction mejorado versión 1 combinado con memoria compartida
}

void reduce(float * d_out, float * d_intermediate, float * d_in, 
            int size, bool usesSharedMemory)
{
  // assumes that size is not greater than maxThreadsPerBlock^2
  // and that size is a multiple of maxThreadsPerBlock
  const int maxThreadsPerBlock = 512;
  int threads = maxThreadsPerBlock;
  int blocks = size / maxThreadsPerBlock;
  if (usesSharedMemory)
  {
      //TODO Invocar version memoria compartida
  }
  else
  {
      //TODO Invocar versión memoria global
  }
  // now we're down to one block left, so reduce it
  threads = blocks; // launch one thread for each block in prev step
  blocks = 1;
  if (usesSharedMemory)
  {
      //TODO Invocar version memoria compartida
  }
  else
  {
      //TODO Invocar versión memoria global
  }
}

int main(int argc, char **argv)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  if (deviceCount == 0) {
      fprintf(stderr, "error: no devices supporting CUDA.\n");
      exit(EXIT_FAILURE);
  }
  int dev = 0;
  cudaSetDevice(dev);

  cudaDeviceProp devProps;
  if (cudaGetDeviceProperties(&devProps, dev) == 0)
  {
      printf("Using device %d:\n", dev);
      printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
              devProps.name, (int)devProps.totalGlobalMem, 
              (int)devProps.major, (int)devProps.minor, 
              (int)devProps.clockRate);
  }

  const int ARRAY_SIZE = 1 << 18;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  // generate the input array on the host
  float *h_in = new float[ARRAY_SIZE];
  float sum = 0.0f;
  for(int i = 0; i < ARRAY_SIZE; i++) {
      // generate random float in [-1.0f, 1.0f]
      h_in[i] = -1.0f + (float)rand()/((float)RAND_MAX/2.0f);
      sum += h_in[i];
  }

  // declare GPU memory pointers
  float * d_in, * d_intermediate, * d_out;

  // allocate GPU memory
  cudaMalloc((void **) &d_in, ARRAY_BYTES);
  cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); 
  cudaMalloc((void **) &d_out, sizeof(float));

  // transfer the input array to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

  // Pasar por el primer parámetro versíon reduction ( 0: memoria global, 1: memoria compartida)
  int whichKernel = 0;
  if (argc == 2) {
      whichKernel = atoi(argv[1]);
  }
        
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // launch the kernel
  switch(whichKernel) {
  case 0:
      printf("Running global reduce\n");
      cudaEventRecord(start, 0);
      reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
      cudaEventRecord(stop, 0);
      break;
  case 1:
      printf("Running reduce with shared mem\n");
      cudaEventRecord(start, 0);
      reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
      cudaEventRecord(stop, 0);
      break;
  default:
      fprintf(stderr, "error: ran no kernel\n");
      exit(EXIT_FAILURE);
  }
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);    
  
  // copy back the sum from GPU
  float h_out;
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

  printf("average time elapsed: %f\n", elapsedTime);

  if( abs(sum-h_out) < 0.001 )
    printf("PASSED \n");
  else
    printf("FAILED \n");

  // free GPU memory allocation
  cudaFree(d_in);
  cudaFree(d_intermediate);
  cudaFree(d_out);
        
  getchar();
  return 0;
}