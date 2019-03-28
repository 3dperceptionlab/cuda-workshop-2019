/// ---------------------------------------------------------------------------
/// CUDA Workshop 2018
/// Universidad de Alicante
/// Práctica 2 - Filtro Mediana
/// Código preparado por: Albert García <agarcia@dtic.ua.es>
///                       Sergio Orts <sorts@ua.es>
/// ---------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

#if _WIN32
	#include <Windows.h>
#else
	#include <sys/types.h>
	#include <sys/time.h>
#endif

// Cabecera necesaria para las rutinas del runtime, es decir, todas
// aquellas que empiezan con cudaXXXXX.
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Cabecera de la librería para tratamiento de bitmaps que usaremos.
#include "EasyBMP.h"

// Tipos y funciones varias para obtener tiempos de ejecución.
// NO MIRAR, DON'T WORRY BE HAPPY
#if _WIN32
	typedef LARGE_INTEGER timeStamp;
	void getCurrentTimeStamp(timeStamp& _time);
	timeStamp getCurrentTimeStamp();
	double getTimeMili(const timeStamp& start, const timeStamp& end);
	double getTimeSecs(const timeStamp& start, const timeStamp& end);
#endif

// Funciones auxiliares
double get_current_time();
void checkCUDAError(const char*);

// Dimensiones de la imagen a procesar
#define WIDTH 1024
#define HEIGHT 1024

// Numero de iteraciones a ejecutar el filtro
#define ITERATIONS 50

// Tamaño de grid y bloque CUDA
#define GRID_W  64
#define GRID_H  64
#define BLOCK_W 16
#define BLOCK_H 16

// Buffers con el halo correspondiente
unsigned char host_input[HEIGHT+2][WIDTH+2];
unsigned char gpu_output[HEIGHT+2][WIDTH+2];
unsigned char host_output[HEIGHT+2][WIDTH+2];

// CUDA kernel filtro mediana - Versión 1D por columnas
// En esta versión, cada hilo procesa una fila de la imagen, iterando sobre
// todas las columnas de la fila que le corresponde.
__global__ void medianFilter1D_col(
	unsigned char *d_output, 
	unsigned char *d_input)
{
	int col, row;
	unsigned char temp;
	int idx, idx_south, idx_north, idx_west, idx_east, idx_north_west, idx_north_east, idx_south_east, idx_south_west;
	int numcols = WIDTH + 2;

	// Calculamos la fila global para este hilo a partir de
	// blockIdx.x, blockDim.x y threadIdx.x.
	// Recuerda sumar 1 para tener en cuenta el halo introducido.
	row = blockIdx.x * blockDim.x + threadIdx.x + 1;

	// Iteramos todas las columnas de la fila correspondiente
	for (col = 1; col <= WIDTH; ++col)
	{
		unsigned char neighborhood[9];

		// Calcular índice lineal para acceder a la fila y 
		// columna correspondiente y sus vecinos.
		idx = row * numcols + col;

		// Calcular indices vecindad 3x3
		idx_south = (row - 1) * numcols + col;
		idx_north = (row + 1) * numcols + col;
		idx_west = row * numcols + (col - 1);
		idx_east = row * numcols + (col + 1);
		idx_north_east = (row + 1) * numcols + (col + 1);
		idx_north_west = (row + 1) * numcols + (col - 1);
		idx_south_east = (row - 1) * numcols + (col + 1);
		idx_south_west = (row - 1) * numcols + (col - 1);
      
		neighborhood[0]= d_input[ idx_south_west ];
		neighborhood[1]= d_input[ idx_south ];
		neighborhood[2]= d_input[ idx_south_east ];
		neighborhood[3]= d_input[ idx_west ];
		neighborhood[4]= d_input[ idx ];
		neighborhood[5]= d_input[ idx_east ];
		neighborhood[6]= d_input[ idx_north_west ];
		neighborhood[7]= d_input[ idx_north ];
		neighborhood[8]= d_input[ idx_north_east ];

		// Ordenar elementos para encontrar la mediana
		for (unsigned int j=0; j<5; ++j)
		{
	        int min=j;
		    for (unsigned int i=j+1; i<9; ++i)
			    if (neighborhood[i] < neighborhood[min])
				    min=i;

			temp=neighborhood[j];
			neighborhood[j]=neighborhood[min];
			neighborhood[min]=temp;
		}

		d_output[idx] = neighborhood[4];
    }
}

// CUDA kernel filtro mediana - Versión 1D por columnas
// En esta versión, cada hilo procesa una columna, iterando sobre todas las
// filas de la columna que le corresponde.
__global__ void medianFilter1D_row(
	unsigned char *d_output, 
	unsigned char *d_input)
{
	int col, row;
	unsigned char temp;
	int idx, idx_south, idx_north, idx_west, idx_east, idx_north_west, idx_north_east, idx_south_east, idx_south_west;
	int numcols = WIDTH + 2;

	// Calculamos la columna global para este hilo a partir de
	// blockIdx.x, blockDim.x y threadIdx.x.
	// Recuerda sumar 1 para tener en cuenta el halo introducido.
	// TODO
	// col = 

	// Iteramos todas las filas de la columna actual
	for (row = 1; row <= HEIGHT; ++row)
	{
		unsigned char neighborhood[9];
		// Calcular índice lineal para acceder a la fila y 
		// columna correspondiente y sus vecinos
		// TODO
		//idx = ;

		// Calcular índices vecindad 3x3
		// TODO 
		/*idx_south = ;
		idx_north = ;
		idx_west = ;
		idx_east = ;
		idx_north_east = ;
		idx_north_west = ;
		idx_south_east = ;
		idx_south_west = ;
      
		neighborhood[0]= d_input[ idx_south_west ];
		neighborhood[1]= d_input[ idx_south ];
		neighborhood[2]= d_input[ idx_south_east ];
		neighborhood[3]= d_input[ idx_west ];
		neighborhood[4]= d_input[ idx ];
		neighborhood[5]= d_input[ idx_east ];
		neighborhood[6]= d_input[ idx_north_west ];
		neighborhood[7]= d_input[ idx_north ];
		neighborhood[8]= d_input[ idx_north_east ];*/

		// Ordenar elementos para encontrar la mediana
		for (unsigned int j=0; j<5; ++j)
		{
			int min=j;
			for (unsigned int i=j+1; i<9; ++i)
				if (neighborhood[i] < neighborhood[min])
					min=i;

			temp=neighborhood[j];
			neighborhood[j]=neighborhood[min];
			neighborhood[min]=temp;
		}

		d_output[idx] = neighborhood[4];
	}
}

// CUDA kernel filtro mediana - Version 2D
// Cada hilo se encargará de procesar un único píxel de la imagen.
__global__ void medianFilter2D(
	unsigned char *d_output,
	unsigned char *d_input)
{
	int col, row;
	unsigned char temp;
	int idx, idx_south, idx_north, idx_west, idx_east, idx_north_west, idx_north_east, idx_south_east, idx_south_west;
	int numcols = WIDTH + 2;

    // Calculamos la fila y columna global para este hilo a partir de
	// blockIdx.x, blockDim.x y threadIdx.x
	// blockIdx.y, blockDim.y y threadIdx.y
	// Recuerda sumar 1 para tener en cuenta el halo introducido  
	// TODO
	// col = ;
	// row = ;

    // Calcular índice lineal para acceder a la fila y 
	// columna correspondiente y sus vecinos.
	unsigned char neighborhood[9];  

	// Calcular índices vecindad 3x3
	idx = row * numcols + col;
	idx_south = (row - 1) * numcols + col;
	idx_north = (row + 1) * numcols + col;
      
	idx_west = row * numcols + (col - 1);
	idx_east = row * numcols + (col + 1);

	idx_north_east = (row + 1) * numcols + (col + 1);
	idx_north_west = (row + 1) * numcols + (col - 1);
	idx_south_east = (row - 1) * numcols + (col + 1);
	idx_south_west = (row - 1) * numcols + (col - 1);
      
	neighborhood[0]= d_input[ idx_south_west ];
	neighborhood[1]= d_input[ idx_south ];
	neighborhood[2]= d_input[ idx_south_east ];
	neighborhood[3]= d_input[ idx_west ];
	neighborhood[4]= d_input[ idx ];
	neighborhood[5]= d_input[ idx_east ];
	neighborhood[6]= d_input[ idx_north_west ];
	neighborhood[7]= d_input[ idx_north ];
	neighborhood[8]= d_input[ idx_north_east ];

	// Ordenar elementos para encontrar la mediana
	for (unsigned int j=0; j<5; ++j)
	{
		int min=j;
		for (unsigned int i=j+1; i<9; ++i)
			if (neighborhood[i] < neighborhood[min])
				min=i;

		temp=neighborhood[j];
		neighborhood[j]=neighborhood[min];
		neighborhood[min]=temp;
	}

	d_output[idx]=neighborhood[4];
}

// Punto de entrada del programa
int main(int argc, char *argv[])
{
	int x, y;
	int i;
	int errors;

	double start_time_inc_data, end_time_inc_data;
	double cpu_start_time, cpu_end_time;

	unsigned char *d_input, *d_output, *d_edge, *tmp;

	unsigned char *input_image;
	unsigned char *output_image;
	int rows;
	int cols;

	// Alojamos memoria en el host para alojar la imagen
	input_image = (unsigned char*)calloc(((HEIGHT * WIDTH) * 1), sizeof(unsigned char));
	// Leemos la imagen 
	BMP Image;
	Image.ReadFromFile("lena_1024_noise.bmp");
	for( int i=0 ; i < Image.TellHeight() ; i++)
		for( int j=0 ; j < Image.TellWidth() ; j++)
			input_image[i*WIDTH+j]=Image(i,j)->Red;
	// Inicializamos a cero el array de CPU para asegurar que el 
	// halo tiene valores correctos
	for (y = 0; y < HEIGHT + 2; y++)
		for (x = 0; x < WIDTH + 2; x++)
			host_input[y][x] = 0;
	// Copiamos la imagen al array de CPU con el halo
	for (y = 0; y < HEIGHT; y++)
		for (x = 0; x < WIDTH; x++)
			host_input[y + 1][x + 1] = input_image[y*WIDTH + x];

	// Calculamos memoria necesaria para alojar la imagen junto con el halo
	// en la memoria de la GPU.
	const int kMemSize = (WIDTH+2) * (HEIGHT+2) * sizeof(unsigned char);

	// Reservamos memoria en la GPU
	cudaMalloc(&d_input, kMemSize);
	cudaMalloc(&d_output, kMemSize);

	// Copiamos todos los arrays a la memoria de la GPU.
	// Tenemos en cuenta dichas transferencias en el tiempo de ejecución.
	start_time_inc_data = get_current_time();

	cudaMemcpy( d_input, host_input, kMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy( d_output, host_input, kMemSize, cudaMemcpyHostToDevice);

	// Aplicamos el filtro mediana un número determinado de iteraciones.
	for (i = 0; i < ITERATIONS; ++i) 
	{
		// Ejecución kernel 1D por filas
		dim3 blocksPerGrid(GRID_H, 1, 1);
		dim3 threadsPerBlock(BLOCK_H, 1, 1);
		//std::cout << "Grid size: (" << blocksPerGrid.x << ", " << blocksPerGrid.y << ", " << blocksPerGrid.z << ")\n";
		//std::cout << "Block size: (" << threadsPerBlock.x << ", " << threadsPerBlock.y << ", " << threadsPerBlock.z << ")\n";
		medianFilter1D_col<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input);

		// Ejecución kernel 1D por columnas
		//TODO - Calcular tamaño de bloque y grid para la correcta ejecucion del kernel
		/*dim3 blocksPerGrid();
		dim3 threadsPerBlock();
		medianFilter1D_row<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input);*/

		// Ejecución kernel 2D
		// TO DO - Calcular tamaño de bloque y grid para la correcta ejecucion del kernel
		/*dim3 blocksPerGrid(,);
		dim3 threadsPerBlock(,);
		medianFilter2D<<< blocksPerGrid, threadsPerBlock >>>(d_output, d_input);*/

		cudaThreadSynchronize();

		// Copiamos en la memoria de la CPU el resultado obtenido
		cudaMemcpy(gpu_output, d_output, kMemSize, cudaMemcpyDeviceToHost);
		// Copiamos el resultado de la GPU hacia la entrada para procesar la siguiente iteración */
		cudaMemcpy( d_input, gpu_output, kMemSize, cudaMemcpyHostToDevice);

		// TODO: Estas copias de memoria se pueden evitar, para ello comenta las
		// transferencias anteriores e intercambia los punteros d_input y d_output
		// para que la salida de esta iteración se convierta en la entrada de la
		// siguiente iteración del filtro mediana.
	}

	// Copiamos el resultado final de la GPU a la CPU
	cudaMemcpy(gpu_output, d_input, kMemSize, cudaMemcpyDeviceToHost);
	end_time_inc_data = get_current_time();

	checkCUDAError("Filtro mediana CUDA: ");

	// Versión CPU
	cpu_start_time = get_current_time();

	unsigned char temp;
	int idx, idx_south, idx_north, idx_west, idx_east, idx_north_west, idx_north_east, idx_south_east, idx_south_west;
	int numcols = WIDTH + 2;
	unsigned char neighborhood[9];

	for (i = 0; i < ITERATIONS; i++)
	{
		for (y = 0; y < HEIGHT; y++)
		{
			for (x = 0; x < WIDTH; x++) 
			{
				neighborhood[0]= host_input[ y+1 -1 ][ x+1 -1 ];
				neighborhood[1]= host_input[ y+1 -1 ][ x+1 ];
				neighborhood[2]= host_input[ y+1 -1][ x+1 +1 ];
				neighborhood[3]= host_input[ y+1 ][ x+1 -1 ];
				neighborhood[4]= host_input[ y+1 ][ x+1 ];
				neighborhood[5]= host_input[ y+1 ][ x+1 +1 ];
				neighborhood[6]= host_input[ y+1+1 ][ x+1 -1 ];
				neighborhood[7]= host_input[ y+1+1 ][ x+1 ];
				neighborhood[8]= host_input[ y+1+1 ][ x+1 +1];

				int j=0;
				// Ordenamos los elementos, solo es necesario ordenar la mitad del array para obtener la mediana
				for (j=0; j<5; ++j)
				{
					// Encontramos el mínimo
					int mini=j;
					for (int l=j+1; l<9; ++l)
					{
							if (neighborhood[l] < neighborhood[mini])
									mini=l;
					}

					temp=neighborhood[j];
					neighborhood[j]=neighborhood[mini];
					neighborhood[mini]=temp;
				}

				host_output[y+1][x+1]=neighborhood[4];
			}
		}

		// Copiamos la salida a la imagen de entrada para una nueva iteración
		for (y = 0; y < HEIGHT; y++)
			for (x = 0; x < WIDTH; x++)
				host_input[y+1][x+1] = host_output[y+1][x+1];
	}

	cpu_end_time = get_current_time();

	// Comprobamos que los resultados de la GPU coinciden con los calculados en la CPU */
	errors = 0;
	for (y = 0; y < HEIGHT; y++)
	{
		for (x = 0; x < WIDTH; x++)
		{
			if ( host_input[y+1][x+1] != gpu_output[y+1][x+1])
			{
				errors++;
				printf("Error en %d,%d (CPU=%i, GPU=%i)\n", x, y, \
					host_output[y+1][x+1], \
					gpu_output[y+1][x+1]);
			}
		}
	}

	if (errors == 0)
		std::cout << "\n\n ***TEST CORRECTO*** \n\n\n";

	// Creamos la imagen de salida y la rellenamos con el resultado ofrecido por la GPU
	output_image = (unsigned char*)calloc(((WIDTH * HEIGHT) * 1), sizeof(unsigned char));

	for (y = 0; y < HEIGHT; y++)
		for (x = 0; x < WIDTH; x++)
			output_image[y*WIDTH+x] = gpu_output[y+1][x+1];

	// Liberamos memoria en el device
	cudaFree(d_input);
	cudaFree(d_output);

	printf("Tiempo ejecución GPU (Incluyendo transferencia de datos): %fs\n", \
		end_time_inc_data - start_time_inc_data);
	printf("Tiempo de ejecución en la CPU                          : %fs\n", \
		 cpu_end_time - cpu_start_time);

	// Copiamos el resultado al formato de la libreria y guardamos el fichero BMP procesado
	for( int i=0 ; i < Image.TellHeight() ; i++)
	{
		for( int j=0 ; j < Image.TellWidth() ; j++)
		{
			Image(i,j)->Red = output_image[i*WIDTH+j];
			Image(i,j)->Green = output_image[i*WIDTH+j];
			Image(i,j)->Blue = output_image[i*WIDTH+j];
		}
	}

	// Guardamos el resultado de aplicar el filtro en un nuevo fichero
	Image.WriteToFile("lena_1024_median.bmp");

	std::cout << "Resultado escrito en lena_1024_median.bmp\n";

	getchar();
	return 0;
}


/* Funciones auxiliares */

#if _WIN32
	void getCurrentTimeStamp(timeStamp& _time)
	{
			QueryPerformanceCounter(&_time);
	}

	timeStamp getCurrentTimeStamp()
	{
			timeStamp tmp;
			QueryPerformanceCounter(&tmp);
			return tmp;
	}

	double getTimeMili()
	{
			timeStamp start;
			timeStamp dwFreq;
			QueryPerformanceFrequency(&dwFreq);
			QueryPerformanceCounter(&start);
			return double(start.QuadPart) / double(dwFreq.QuadPart);
	}
#endif 

double get_current_time()
{
	#if _WIN32 
		return getTimeMili();
	#else
		static int start = 0, startu = 0;
		struct timeval tval;
		double result;

		if (gettimeofday(&tval, NULL) == -1)
			result = -1.0;
		else if(!start) {
			start = tval.tv_sec;
			startu = tval.tv_usec;
			result = 0.0;
		}
		else
			result = (double) (tval.tv_sec - start) + 1.0e-6*(tval.tv_usec - startu);
		return result;
	#endif
}

/* Función para comprobar errores CUDA */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
	getchar();
        exit(EXIT_FAILURE);
    }                         
}

