#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <iostream>
#include <iterator>
#include <algorithm>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

// This example illustrates how to implement the SAXPY
// operation (Y[i] = a * X[i] + Y[i]) using Thrust. 
// The saxpy_slow function demonstrates the most
// straightforward implementation using a temporary
// array and two separate transformations, one with
// multiplies and one with plus.  The saxpy_fast function
// implements the operation with a single transformation
// and represents "best practice".

struct saxpy_functor : public thrust::binary_function<float,float,float>
{
    const float a;

    //hacer constructor que reciba la constante escalar A;


    //implementar el operator()
    __host__ __device__
        float operator()(const float& x, const float& y) const { 
            //hacer operacion!
        }
};

void saxpy_fast(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
    // operación thrust con functor que realice Y <- A * X + Y
    thrust::transform(......., saxpy_functor(....));
}

void saxpy_slow(float A, thrust::device_vector<float>& X, thrust::device_vector<float>& Y)
{
   //creamos vector temporal
    thrust::device_vector<float> temp(X.size());
   
    // Rellenamos el vector temp con los valores de A ; temp <- A
    thrust::fill(....);
    
    // Mediante una operacion transform realizamos la multiplicación; temp <- A * X
    thrust::transform(......., thrust::multiplies<float>());

    // Por ultimo mediante otra operacion transform sumamos el vector temp y el vector Y; Y <- A * X + Y
    thrust::transform(..........., thrust::plus<float>());
}

int main(void)
{

   // Declaramos los eventos y los inicializamos
    cudaEvent_t     start, stop;
    

    // Vectores de la parte host
    const int N = 1024*1024;
    std::vector<float> x(N);
    std::vector<float> y(N);
   
	for(int i=0;i<N;i++)
	{
		x[i]=i;
		y[i]=5*i;
	}

    {
       cudaEventCreate( &start );
       cudaEventCreate( &stop );
       cudaEventRecord( start, 0);
       
        // Inicializar los device_vector X e Y con tamaño N con los valores del vector x e y respectivamente
        thrust::device_vector<float> X(.........);
        thrust::device_vector<float> Y(........);

        // Llamada al metodo mas lento
        saxpy_slow(2.0, X, Y);
        
        // Detenemos los eventos y mostramos los tiempos
       cudaEventRecord( stop, 0 );
       cudaEventSynchronize( stop );
       float   elapsedTime;
       cudaEventElapsedTime( &elapsedTime,
                                           start, stop );
       printf( "Time consumido saxpy_slow:  %3.1f ms\n", elapsedTime );
    }
    
    

    {
       cudaEventCreate( &start );
       cudaEventCreate( &stop );
       cudaEventRecord( start, 0 );
       
        // Inicializar los device_vector X e Y con tamaño N con los valores del vector x e y respectivamente
        thrust::device_vector<float> X(.....);
        thrust::device_vector<float> Y(.....);

        // llamada al metodo rapido
        saxpy_fast(2.0, X, Y);
        
       // Detenemos los eventos y mostramos los tiempos
       cudaEventRecord( stop, 0 ) ;
       cudaEventSynchronize( stop ) ;
       float   elapsedTime;
       cudaEventElapsedTime( &elapsedTime,
                                           start, stop ) ;
       printf( "Time consumido saxpy_fast:  %3.1f ms\n", elapsedTime );
    }
    
    return 0;
}
