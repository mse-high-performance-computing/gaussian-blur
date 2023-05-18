
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

	float red = 0;
	float green = 0;
	float blue = 0;

	// Apply gauss kernel
	for (int i = 0; i < (*smoothKernelDimension); i++) {
		int kY = y + (i - (*smoothKernelDimension/2));
		// Border handling, use nearest valid pixel
		if (kY < 0 || kY >= (*height)) 
			kY = y;
		
		for (int j = 0; j < (*smoothKernelDimension); j++) {
			int kX = x + (j - (*smoothKernelDimension/2));
			// Border handling, use nearest valid pixel
			if (kX < 0 || kX >= (*width)) 
				kX = x;
			
			size_t index = channels * (kY * (*width) + kX);
			size_t kernelIndex = (i * (*smoothKernelDimension) + j);
			red += A[index] * smoothKernel[kernelIndex];
			green += A[index + 1] * smoothKernel[kernelIndex];
			blue += A[index + 2] * smoothKernel[kernelIndex];
		}
	}

	// Write results for each color component
	// Three consecutive color components represent one pixel
	size_t index = channels * (y * (*width) + x);
	B[index] = red;
	B[index + 1] = green;
	B[index + 2] = blue;
}

