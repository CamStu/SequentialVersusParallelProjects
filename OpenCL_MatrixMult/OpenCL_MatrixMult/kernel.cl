__kernel void mult_mat(__global const float *A, __global const float *B, __global float *C) {

	// Get the index of the current element to be processed
	int col = get_global_id(0);
	int row = get_global_id(1);

	float val;

	for (int i = 0; i < 100; i++) {
		val += A[row * 100 + i] * B[i * 100 + col];
	}
	C[row * 100 + col] = val;
}