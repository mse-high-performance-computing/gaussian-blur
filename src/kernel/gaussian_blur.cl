
__kernel void gaussian_blur(
	__global const uchar *A,
	__global uchar *B,
	__constant int *width,
	__constant int *height,
	__constant float *smoothKernel,
	__constant int *smoothKernelDimension,
	__constant bool *horizontal
)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t channels = 3;

	size_t N = (*smoothKernelDimension) * (*smoothKernelDimension);

	size_t index = 3 * (y * (*width) + x);

	float red = 0;
	float green = 0;
	float blue = 0;

	if (*horizontal) {
		for (int i = 0; i < N; i++) {
			int kX = x + (i - (N/2));
			if (kX < 0 || kX >= (*width)) 
				kX = x;
			
			red += A[3 * (y * (*width) + kX)]
			* smoothKernel[i];
			green += A[3 * (y * (*width) + kX) + 1]
			* smoothKernel[i];
			blue += A[3 * (y * (*width) + kX) + 2]
			* smoothKernel[i];
		}
	} else {
		for (int i = 0; i < N; i++) {
			int kY = y + (i - (N/2));
			if (kY < 0 || kY >= (*height)) 
				kY = y;
			
			red += A[3 * (kY * (*width) + x)] 
			* smoothKernel[i];
			green += A[3 * (kY * (*width) + x) + 1] 
			* smoothKernel[i];
			blue += A[3 * (kY * (*width) + x) + 2]  
			* smoothKernel[i];
		}
	}

	B[index] = red;
	B[index+1] = green;
	B[index+2] = blue;
}

