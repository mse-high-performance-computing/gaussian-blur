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
	__constant int *height,
	__constant float *smoothKernel,
	__constant int *smoothKernelDimension
)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t channels = 3;

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
		printf("dim %i\n", *smoothKernelDimension);
		printf("---\n");
	}

	float red = 0;
	float green = 0;
	float blue = 0;

	for (int i = 0; i < (*smoothKernelDimension); i++) {
		int kY = y + (i - (*smoothKernelDimension/2));
		if (kY < 0 || kY >= (*height)) 
			kY = y;
		
		for (int j = 0; j < (*smoothKernelDimension); j++) {
			int kX = x + (j - (*smoothKernelDimension/2));
			if (kX < 0 || kX >= (*width)) 
				kX = x;
			
			red += A[3 * (kY * (*width) + kX)] 
			* smoothKernel[(i * (*smoothKernelDimension) + j)];
			green += A[3 * (kY * (*width) + kX) + 1] 
			* smoothKernel[(i * (*smoothKernelDimension) + j)];
			blue += A[3 * (kY * (*width) + kX) + 2]  
			* smoothKernel[(i * (*smoothKernelDimension) + j)];
		}
	}

	// red /= (*smoothKernelDimension);
	// blue /= (*smoothKernelDimension);
    // green /= (*smoothKernelDimension);


	B[index] = red;
	B[index+1] = green;
	B[index+2] = blue;
}

