/*
* a kernel that add the elements of two vectors pairwise
*/
__kernel void vector_add(
	__global const int *A,
	__global int *B)
{
	size_t i = get_global_id(0);
	// Number of global work-items
	size_t size = get_global_size(0);

	if (i < size-1) {
		B[i] = A[i] + A[i+1];
	} else {
		B[i] = A[i];
	}
}

__kernel void gaussian_blur(
	__global const uchar *A,
	__global uchar *B,
	__constant int *width,
	__constant int *height
)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t channels = 3;

	// printf("x %i, y %i, width %i, height %i \n", x, y, *width, *height);

	size_t index = 3 * (y * (*width) + x);
	if (x == 255 && y == 511) {
		// printf("x %i, y %i, index %i, width %i, height %i", x, y, index, width, height);
		int baum = *width;
		int baum2 = *height;
		printf("---\n");
		printf("index %i\n", index);
		printf("index %i\n", A[index]);
		printf("x %i\n", x);
		printf("y %i\n", y);
		printf("width %i\n", baum);
		printf("height %i\n", baum2);
		printf("---\n");
	}
	
	// https://registry.khronos.org/OpenCL/specs/3.0-unified/html/OpenCL_C.html#operators-prepost
	// Does not seem to work?
	// B[index] = A[index++];
	// B[index] = A[index++];
	// B[index] = A[index++];

	B[index] = A[index];
	B[index+1] = A[index+1];
	B[index+2] = A[index+2];
}

