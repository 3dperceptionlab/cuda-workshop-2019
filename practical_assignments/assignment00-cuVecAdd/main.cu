/// ---------------------------------------------------------------------------
/// CUDA Workshop 2019
/// Universidad de Alicante
/// Práctica 0 - Suma de Vectores
/// Código preparado por: Albert García <agarcia@dtic.ua.es>
///                       Sergio Orts <sorts@ua.es>
/// ---------------------------------------------------------------------------

#include <iostream>
// Cabecera necesaria para las rutinas del runtime de CUDA (cudaFree, cudaMalloc...)
#include <cuda_runtime.h>
// Cabecera necesaria para variables y tipos de CUDA...
#include <device_launch_parameters.h>

/// Kernel para suma de vectores.
/// Este kernel computará la suma de dos vectores de forma que cada
/// hilo será responsable de sumar un elemento de dichos vectores.
__global__ 
void suma_vectores(
	const float *cpA, 
	const float *cpB, 
	float *pC, 
	const int cNumElements)
{
	/// === PASO 2 ============================================================
	/// Define los índices del elemento a ser sumado por cada hilo empleando las
	/// variables de CUDA: threadIdx, blockIdx y blockDim.
	/// TODO:
	int idx_ = ???;

	/// Suma las dos posiciones en el vector de salida, cada hilo debe computar
	/// el cálculo de un elemento.
	/// TODO:

	/// === FIN PASO 2 ========================================================
}

/// Kernel para suma de vectores con stride.
/// Este kernel computará la suma de dos vectores de forma que cada
/// hilo será responsable de sumar varios elementos de dichos vectores.
__global__
void suma_vectores_strided(
	const float *cpA,
	const float *cpB,
	float *pC,
	const int cNumElements)
{
	/// === PASO 4 ============================================================
	/// Modifica el kernel anterior para que se puedan sumar vectores de un
	/// tamaño muy grande. Recuerda cambiar los parámetros de invocación y
	/// llamar a este kernel en lugar de al anterior.
	int idx_ = ???;

	/// Sumar las posiciones adecuadas en el vector de salida, cada hilo debe
	/// computar más de un elemento.
	/// TODO:

	/// === FINAL PASO 4 ======================================================
}

int main(void)
{
	// Elegimos la GPU a utilizar, en este caso la 0
	cudaSetDevice(0);

	// Calculamos el tamaño en bytes del vector
	/// === PASO 3 ============================================================
	/// Modifica el número de elementos a sumar.
  const int kNumElements = 25600;
	/// === FIN PASO 3 ========================================================
  size_t vector_size_bytes_ = kNumElements * sizeof(float);
	std::cout << "[Vector addition of " << kNumElements << " elements]\n";

  // Reservamos memoria para los vectores en el HOST
  float *h_A_ = (float *)malloc(vector_size_bytes_);
  float *h_B_ = (float *)malloc(vector_size_bytes_);
  float *h_C_ = (float *)malloc(vector_size_bytes_);

  // Comprobamos que las reservas se han efectuado correctamente
  if (h_A_ == NULL || h_B_ == NULL || h_C_ == NULL)
  {
		std::cerr << "Failed to allocate host vectors!\n";
		getchar();
    exit(-1);
  }

  // Inicializamos los vectores en el HOST con valores arbitrarios
  for (int i = 0; i < kNumElements; ++i)
  {
		h_A_[i] = rand()/(float)RAND_MAX;
		h_B_[i] = rand()/(float)RAND_MAX;
  }

	// Reservamos memoria para los vectores en el DEVICE
  float *d_A_ = NULL;
	float *d_B_ = NULL;
	float *d_C_ = NULL;

  cudaMalloc((void **)&d_A_, vector_size_bytes_);
	cudaMalloc((void **)&d_B_, vector_size_bytes_);
	cudaMalloc((void **)&d_C_, vector_size_bytes_);

  // Copiamos los vectores A y B de HOST a DEVICE
	std::cout << "Copy input data from the host memory to the CUDA device\n";

  cudaMemcpy(d_A_, h_A_, vector_size_bytes_, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B_, h_B_, vector_size_bytes_, cudaMemcpyHostToDevice);

  // Lanzamos el kernel de suma de vectores y comprobamos errores
	/// === PASO 1 ============================================================
	/// Establece los parámetros de invocación del kernel e invócalo.
  int threads_per_block_ = 256;
	int blocks_per_grid_ = ???;
	/// === PASO 3 ============================================================
	/// Modifica el cálculo del tamaño del grid para que se puedan sumar
	/// vectores de un tamaño arbitrario.
	/// blocks_per_grid_ = ???;
	/// === FIN PASO 3 ========================================================
	/// === PASO 4 ============================================================
	/// Establece los parámetros de invocación del kernel modificado.
	/// blocks_per_grid_ = ???;
	/// === FIN PASO 4 ========================================================

	dim3 block(threads_per_block_, 1, 1);
	dim3 grid(blocks_per_grid_, 1, 1);

	std::cout << "CUDA kernel launch with " << blocks_per_grid_ << " blocks of " << threads_per_block_ << " threads\n";
  suma_vectores<<<???, ???>>>(???);
	cudaError_t err_ = cudaGetLastError();

  if (err_ != cudaSuccess)
  {
		std::cerr << "Failed to launch sumaVectores kernel (error code " << cudaGetErrorString(err_) << ")!\n";
		getchar();
    exit(-1);
  }
	/// === FIN PASO 1 ========================================================

  // Copiamos el vector resultante del DEVICE al HOST
	std::cout << "Copy output data from the CUDA device to the host memory\n";

  cudaMemcpy(h_C_, d_C_, vector_size_bytes_, cudaMemcpyDeviceToHost);

  // Verificamos el resultado
  for (int i = 0; i < kNumElements; ++i)
  {
		// Dado que utilizamos floats las comparaciones de igualdad fallarían
		// por el orden de las operaciones por lo que utilizamos una comparación
		// con un umbral 1e-5
    if (fabs(h_A_[i] + h_B_[i] - h_C_[i]) > 1e-5)
    {
			std::cerr << "Result verification failed at element " << i << "!\n";
			getchar();
      exit(-1);
    }
  }

	std::cout << "Test PASSED\n";

  // Liberamos la memoria en el DEVICE
  cudaFree(d_A_);
  cudaFree(d_B_);
	cudaFree(d_C_);

  // Liberamos la memoria en el HOST
  free(h_A_);
  free(h_B_);
  free(h_C_);

	// Reiniciamos el dispositivo
	// cudaDeviceReset hace que el driver limpie todo estado actual. Aunque no es
	// una operación obligatoria, es una buena práctica. Además, es necesaria si
	// estamos realiando profiling de la aplicación.
  cudaDeviceReset();

	// Finalizamos el programa
	std::cout << "Done\n";
	getchar();
  return 0;
}

