/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include "./common/book.h"
#include "./common/cpu_bitmap.h"
#include "cuda.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define DIM 1000

/* retorna "a - b" en segundos - RUTINA TIEMPOS WINDOWS */
double performancecounter_diff(LARGE_INTEGER *a, LARGE_INTEGER *b)
{
  LARGE_INTEGER freq;
  QueryPerformanceFrequency(&freq);
  return (double)(a->QuadPart - b->QuadPart) / (double)freq.QuadPart;
}

struct cuComplex {
    float   r;
    float   i;
    __host__ __device__ cuComplex( float a, float b ) : r(a), i(b)  {}
    __host__ __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __host__ __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __host__ __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r+a.r, i+a.i);
    }
};

__host__ __device__ int julia( int x, int y ) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM/2 - x)/(DIM/2);
    float jy = scale * (float)(DIM/2 - y)/(DIM/2);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i=0; i<200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

__global__ void kernel( unsigned char *ptr ) {	
	
	// COMPLETAR!
	//
	int x = //TODO
	int y = //TODO
	int offset = y * (blockDim.x*gridDim.x) + x;

    // now calculate the value at that position
    int juliaValue = julia( x, y );
    ptr[offset*4 + 0] = 255 * juliaValue;
    ptr[offset*4 + 1] = 0;
    ptr[offset*4 + 2] = 0;
    ptr[offset*4 + 3] = 255;
}

void kernel_CPU( unsigned char *ptr ){
    for (int y=0; y<DIM; y++) {
        for (int x=0; x<DIM; x++) {
            int offset = x + y * DIM;

            int juliaValue = julia( x, y );
            ptr[offset*4 + 0] = 255 * juliaValue;
            ptr[offset*4 + 1] = 0;
            ptr[offset*4 + 2] = 0;
            ptr[offset*4 + 3] = 255;
        }
    }
 }

// globals needed by the update routine
struct DataBlock {
    unsigned char   *dev_bitmap;
};

int main( void ) {
    DataBlock   data;
    CPUBitmap bitmap( DIM, DIM, &data );
    unsigned char    *dev_bitmap;

	// Declaramos los eventos y los inicializamos
    cudaEvent_t     start, stop;
    HANDLE_ERROR( cudaEventCreate( &start ) );
    HANDLE_ERROR( cudaEventCreate( &stop ) );
    HANDLE_ERROR( cudaEventRecord( start, 0 ) );

    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    data.dev_bitmap = dev_bitmap;
    
	// COMPLETAR
	// TODO
	dim3    grid();
	dim3	block();
	kernel<<< >>>( dev_bitmap );
    
	HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );
    
	// Detenemos los eventos y mostramos los tiempos
  HANDLE_ERROR( cudaEventRecord( stop, 0 ) );
  HANDLE_ERROR( cudaEventSynchronize( stop ) );
  float   elapsedTime;
  HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime,
                                        start, stop ) );
  printf( "Time consumido generacion fractal GPU:  %3.1f ms\n", elapsedTime );
	HANDLE_ERROR( cudaEventDestroy( start ) );
  HANDLE_ERROR( cudaEventDestroy( stop ) );
	HANDLE_ERROR( cudaFree( dev_bitmap ) );
      
	// Calculo CPU
	LARGE_INTEGER t_ini, t_fin;
	CPUBitmap bitmap2( DIM, DIM );
	unsigned char *ptr = bitmap2.get_ptr();

	QueryPerformanceCounter(&t_ini);	
  kernel_CPU( ptr );
  QueryPerformanceCounter(&t_fin);

  double cpu_runtime = performancecounter_diff(&t_fin, &t_ini);
  printf( "Time consumido generacion fractal CPU:  %3.1f ms\n", cpu_runtime*1000);

  bitmap.display_and_exit();
}

