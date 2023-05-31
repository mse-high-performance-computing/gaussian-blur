
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
	size_t index = channels * (y * (*width) + x);

	size_t localX = get_local_id(0);
	size_t localY = get_local_id(1);
	size_t localXY = *horizontal ? localX : localY;
	size_t localIndex = channels * (*horizontal ? localX : localY);
	size_t localMax = (*horizontal ? *width : *height)-1;

	float red = 0;
	float green = 0;
	float blue = 0;

	// Minimize access of source pixel values
	pixel[localIndex] = A[index];
	pixel[localIndex + 1] = A[index + 1];
	pixel[localIndex + 2] = A[index + 2];
	barrier(CLK_LOCAL_MEM_FENCE);

	// Apply gauss kernel
	for (int i = 0; i < (*smoothKernelDimension); i++) {
		int k = localXY + (i - ((*smoothKernelDimension)/2));
		// Border handling, use nearest valid pixel
		k = clamp(k, 0, (int)localMax);

		size_t kIndex = channels * k;
		float kernelValue = smoothKernel[i];
		red += pixel[kIndex] * kernelValue;
		green += pixel[kIndex + 1] * kernelValue;
		blue += pixel[kIndex + 2] * kernelValue;
	}

	// Write results for each color component
	// Three consecutive color components represent one pixel
	B[index] = red;
	B[index + 1] = green;
	B[index + 2] = blue;
}
