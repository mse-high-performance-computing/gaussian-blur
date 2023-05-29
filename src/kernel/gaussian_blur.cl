
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

	size_t localX = get_local_id(0);
	size_t localY = get_local_id(1);

	float red = 0;
	float green = 0;
	float blue = 0;

	// Minimize access of source pixel values
	size_t index = channels * (y * (*width) + x);
	if (*horizontal) {
		size_t localIndex = channels * localX;
		pixel[localIndex] = A[index];
		pixel[localIndex + 1] = A[index + 1];
		pixel[localIndex + 2] = A[index + 2];
	} else {
		size_t localIndex = channels * localY;
		pixel[localIndex] = A[index];
		pixel[localIndex + 1] = A[index + 1];
		pixel[localIndex + 2] = A[index + 2];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Apply gauss kernel
	if (*horizontal) {
		for (int i = 0; i < (*smoothKernelDimension); i++) {
			int kX = localX + (i - ((*smoothKernelDimension)/2));
			// Border handling, use nearest valid pixel
			if (kX < 0 || kX >= (*width)) 
				kX = x;
			
			size_t index = channels * kX;
			red += pixel[index] * smoothKernel[i];
			green += pixel[index + 1] * smoothKernel[i];
			blue += pixel[index + 2] * smoothKernel[i];
		}
	} else {
		for (int i = 0; i < (*smoothKernelDimension); i++) {
			int kY = localY + (i - ((*smoothKernelDimension)/2));
			// Border handling, use nearest valid pixel
			if (kY < 0 || kY >= (*height)) 
				kY = y;
			
			size_t index = channels * kY;
			red += pixel[index] * smoothKernel[i];
			green += pixel[index + 1] * smoothKernel[i];
			blue += pixel[index + 2] * smoothKernel[i];
		}
	}

	// Write results for each color component
	// Three consecutive color components represent one pixel
	B[index] = red;
	B[index + 1] = green;
	B[index + 2] = blue;
}

