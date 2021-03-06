#include <iostream>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

// Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
	for (int i = 0; i < n; i++)
		y[i] = x[i] + y[i];
}

//Multiplies a,b stores result in c
__global__
void MatrixMult(float *a, float *b, float *c) {
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	float val = 0;

	if (row > 100 || col > 100) 
		return;

	for (int i = 0; i < 100; i++) {
		val += a[row * 100 + i] * b[i * 100 + col];
	}
	c[row * 100 + col] = val;
}


int main(void)
{
	int width = 100, height = 100; //100x100
	float *x, *y, *z;

	cudaEvent_t start, stop;
	float time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Allocate Unified Memory � accessible from CPU or GPU
	// While 1D arrays, these will be accessed by converting 2D index space to linear space
	cudaMallocManaged(&x, width*height * sizeof(float));	
	cudaMallocManaged(&y, width*height * sizeof(float));
	cudaMallocManaged(&z, width*height * sizeof(float));

	//Initialize arrays on host
	for(int i=0; i < width; i++)
		for (int j = 0; j < height; j++) {
			x[i*width + j] = 1.0f;
			y[i*width + j] = 2.0f;
		}
	//A 10,000x10,000 Product matrix could be computer with one thread per entry with 100x100 blocks with 100x100 threads each
	dim3 blockSize(10, 10);
	dim3 gridSize(10, 10);

	cudaEventRecord(start, 0);
	MatrixMult <<< gridSize, blockSize >>> (x, y, z);	//Block and thread dimensions chosen to be within hardware constraints
	
	//Timing information
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	printf("Operation complete\n");
	printf("Elapsed time on GPU= %f ms \n", time);

	// Free memory
	cudaFree(x);
	cudaFree(y);
	cudaFree(z);

	return 0;
}