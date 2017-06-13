#include <iostream>
#include <math.h>
#include <time.h>

/* For now...
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
*/


float dotprod(int row, int col, float **a, float **b) {
	float val = 0;
	for (int i = 0; i < 100; i++)
		val += a[row][i] * b[i][col];
	return val;
}

//Multiplies a,b stores result in c
void MatrixMult(float **a, float **b, float **c) {
	for (int i = 0; i<100; i++)
		for (int j = 0; j < 100; j++) {
			c[i][j] = dotprod(i, j, a, b);
		}
}

const int width = 100;
const int height = 100;

int main(void)
{
	//	int N = 1 << 20;	//Million elements
	//int width = 1000, height = 1000; //10,000x10,000

	float **x, **y, **z;
	clock_t start, end;
	double time = 0;

	// 2D arrays 
	x = (float**)malloc(width * sizeof(float*));
	y = (float**)malloc(width * sizeof(float*));
	z = (float**)malloc(width * sizeof(float*));
	for (int i = 0; i < height; i++) {
		x[i] = (float*)malloc(height * sizeof(float));
		y[i] = (float*)malloc(height * sizeof(float));
		z[i] = (float*)malloc(height * sizeof(float));
	}
	//Initialize arrays on host
	for (int i = 0; i < width; i++)
		for (int j = 0; j < height; j++) {
			x[i][j] = 1.0f;
			y[i][j] = 2.0f;
		}

	/*
	//A 10,000x10,000 Product matrix could be computer with one thread per entry with 100x100 blocks with 100x100 threads each
	dim3 blockSize(100, 100);
	dim3 gridSize(100, 100);
	MatrixMult << < gridSize, blockSize >> > (x, y, z);	//Block and thread dimensions chosen to be within hardware constraints
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	*/

	start = clock();
	MatrixMult(x, y, z);
	//Timing information
	end = clock();
	time = ((float)(end - start) / CLOCKS_PER_SEC);
	printf("Operation complete\n");
	printf("Time elapsed on CPU= %f", time);
	printf(" seconds\n");
	printf("zero index of z= %f\n", z[0][0]);

	// Free memory
	free(x);
	free(y);
	free(z);

	return 0;
}