/*
	Adapting code from https://github.com/smistad/OpenCL-Getting-Started/
*/
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

const int width = 100;
const int height = 100;

int main(void) {
	printf("started running\n");

	// Create the two input vectors
	int i;

	//	Create arrays
	float *x, *y, *z;

	x = (float*)malloc(sizeof(float)*width*height);
	y = (float*)malloc(sizeof(float)*width*height);
	z = (float*)malloc(sizeof(float)*width*height);

	for(int i=0; i<width; i++)
		for (int j = 0; j < height; j++) {
			x[i*width + j] = 1.f;
			y[i*width + j] = 2.f;
		}
	
	// Load the kernel source code into the array source_str
	FILE *fp;
	char *source_str;
	size_t source_size;
	
	fp = fopen("kernel.cl", "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	// Get platform and device information, choosing the first device
	cl_device_id device_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;

	cl_int ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
	cl_platform_id *platforms = NULL;
	platforms = (cl_platform_id*)malloc(ret_num_platforms * sizeof(cl_platform_id));

	ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);

	ret = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 1,
		&device_id, &ret_num_devices);

	// Create an OpenCL context
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

	// Create a command queue
	cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

	// Create memory buffers on the device for each vector 
	cl_mem x_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
		width*height * sizeof(float), NULL, &ret);
	cl_mem y_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
		width*height * sizeof(float), NULL, &ret);
	cl_mem z_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
		width*height * sizeof(float), NULL, &ret);

	// Copy the lists A and B to their respective memory buffers
	ret = clEnqueueWriteBuffer(command_queue, x_mem, CL_TRUE, 0,
		width*height * sizeof(float), x, 0, NULL, NULL);

	ret = clEnqueueWriteBuffer(command_queue, y_mem, CL_TRUE, 0,
		width*height * sizeof(float), y, 0, NULL, NULL);

	// Create a program from the kernel source
	cl_program program = clCreateProgramWithSource(context, 1,
		(const char **)&source_str, (const size_t *)&source_size, &ret);

	// Build the program
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	
	// Create the OpenCL kernel
	cl_kernel kernel = clCreateKernel(program, "mult_mat", &ret);

	// Set the arguments of the kernel
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&x_mem);
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&y_mem);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&z_mem);
	
	// Execute the OpenCL kernel on the list
	//size_t global_item_size = LIST_SIZE; // Process the entire lists
	//size_t local_item_size = 64; // Divide work items into groups of 64
							
	size_t global_size[] = { width,height };
	//size_t group_pattern[] = {100,100};
	size_t local_size[] = {10,10};
	//Execute kernel
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL,
		&global_size, &local_size, 0, NULL, NULL);
	
	// Read the memory buffer C on the device to the local variable z
	ret = clEnqueueReadBuffer(command_queue, z_mem, CL_TRUE, 0,
		width*height * sizeof(float), z, 0, NULL, NULL);
	
	printf("zero index of z= %f\n", z[0]);
	//printf("1000 index of z= %f\n", z[1000]);

	// Clean up
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(x_mem);
	ret = clReleaseMemObject(y_mem);
	ret = clReleaseMemObject(z_mem);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	free(x);
	free(y);
	free(z);
	return 0;
}