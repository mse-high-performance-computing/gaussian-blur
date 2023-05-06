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