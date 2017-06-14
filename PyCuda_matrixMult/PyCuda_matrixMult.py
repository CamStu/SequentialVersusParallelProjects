#!
import numpy as np
from pycuda import driver, compiler, gpuarray, tools

import pycuda.autoinit

kernel_code = """
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
"""
#Create the matrices, default data type will be floats for these
a = np.ones(100*100)
b = np.full(100*100,2.0)
c = np.empty(100*100)

agpu = gpuarray.to_gpu(a)
bgpu = gpuarray.to_gpu(b)
cgpu = gpuarray.to_gpu(c)

#Compile kernel
mod = compiler.SourceModule(kernel_code)

#Getting the function after it has been compiled
MatrixMult = mod.get_function("MatrixMult")

grid = (10,10)
block =(10,10)
#Call kernel function
MatrixMult(apu,bgpu,cgpu, block=block, grid=grid )

print("All done")
