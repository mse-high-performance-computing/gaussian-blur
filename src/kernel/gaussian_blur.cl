
__kernel void gaussian_blur(
	__global const uchar *A,
	__global uchar *B,
	__constant int *width,
	__constant int *height,
	__constant float *smoothKernel,
	__constant int *smoothKernelDimension,
	__constant bool *horizontal,
	__local uchar* pixel
)
{
	size_t x = get_global_id(0);
	size_t y = get_global_id(1);
	size_t channels = 3;

	float red = 0;
	float green = 0;
	float blue = 0;

	// Apply gauss kernel
	if (*horizontal) {
		for (int i = 0; i < (*smoothKernelDimension); i++) {
			int kX = x + (i - ((*smoothKernelDimension)/2));
			// Border handling, use nearest valid pixel
			if (kX < 0 || kX >= (*width)) 
				kX = x;
			
			size_t index = channels * (y * (*width) + kX);
			red += A[index] * smoothKernel[i];
			green += A[index + 1] * smoothKernel[i];
			blue += A[index + 2] * smoothKernel[i];
		}
	} else {
		for (int i = 0; i < (*smoothKernelDimension); i++) {
			int kY = y + (i - ((*smoothKernelDimension)/2));
			// Border handling, use nearest valid pixel
			if (kY < 0 || kY >= (*height)) 
				kY = y;
			
			size_t index = channels * (kY * (*width) + x);
			red += A[index] * smoothKernel[i];
			green += A[index + 1] * smoothKernel[i];
			blue += A[index + 2] * smoothKernel[i];
		}
	}

	// Write results for each color component
	// Three consecutive color components represent one pixel
	size_t index = channels * (y * (*width) + x);
	B[index] = red;
	B[index + 1] = green;
	B[index + 2] = blue;
}

