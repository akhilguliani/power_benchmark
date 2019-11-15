// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void print_matrix(const float *A, int nr_rows_A, int nr_cols_A) {

	for(int i = 0; i < nr_rows_A; ++i){
		for(int j = 0; j < nr_cols_A; ++j){
			std::cout << A[j * nr_rows_A + i] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

int main() {

	cudaStream_t copyStream;
	cudaError_t result = cudaStreamCreate(&copyStream);

	// Allocate the src on CPU
	long SIZE = 512*1024*1024;
	int* src = (int*) malloc(SIZE * sizeof(int));
	// int* src; 
	// cudaMallocHost((void**) &src, SIZE * sizeof(int));
	for (int i = 0; i < SIZE ; ++i) {
		src[i] = 5;
	}

	// Allocate DST on gpu	
	int* dst;
	cudaMalloc(&dst, SIZE*sizeof(int));

	cudaMemcpyAsync((void*)dst, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice, copyStream);

	// Create a handle for CUBLAS	
	cudaDeviceSynchronize();

	for (int j = 0 ; j < 100; j++){
		cudaMemcpyAsync((void*)src, (void*)dst, sizeof(int) * SIZE, cudaMemcpyDeviceToHost, copyStream);
	} 
	cudaStreamSynchronize(copyStream);

	//Free pinned memory
	cudaFreeHost(src);
	cudaFree(dst);


	result = cudaStreamDestroy(copyStream);
	return 0;
}
