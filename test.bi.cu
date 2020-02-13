// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <cmath>

// Randomization helpers 
void init_sin(float *A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1){

    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = sin(i + j * lda + i_batch * stride);

}


void init_cos(float *A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1){

    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = cos(i + j * lda + i_batch * stride);

}

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

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


void setup_nvlink(){
	int numGPUs = 4;

	int i = 0;
	for (int j = 0; j < numGPUs; j++) {
		int access = 0;
		cudaDeviceCanAccessPeer(&access, i, j);
		if (access) {
			printf("Enabling %d to %d\n", i, j);
			cudaSetDevice(i);
			cudaCheckError();
			cudaDeviceEnablePeerAccess(j, 0);
			cudaCheckError();
			cudaSetDevice(j);
			cudaCheckError();
			cudaDeviceEnablePeerAccess(i, 0);
			cudaCheckError();
			cudaSetDevice(i);
			cudaCheckError();
		}
		fflush(stdout);
	}
}



int main(int argc, char* argv[]) {
        
	if (argc != 4){
		std::cout << "USAGE: " << argv[0] <<"<size> <reps> <active-links>" <<std::endl ;
		exit(-1);
	}

	int multiplier = atoi(argv[1]);

	int reps = atoi(argv[2]);
	
	int nlinks = atoi(argv[3]);
        
        setup_nvlink();

	cudaSetDevice(0);
	cudaStream_t copyStream, copyStream2;
	cudaStream_t copyStream3, copyStream4;
	cudaError_t result = cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&copyStream2, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&copyStream3, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&copyStream4, cudaStreamNonBlocking);

	// Allocate the src on CPU
	long SIZE = multiplier*1024*1024;
	// int* src = (int*) malloc(SIZE * sizeof(int));
	int* src; 
	cudaMallocHost((void**) &src, SIZE * sizeof(int));

	int* src_h; 
	cudaMallocHost((void**) &src_h, SIZE * sizeof(int));

	for (int i = 0; i < SIZE ; ++i) {
		src[i] = (unsigned int)-1;
		src_h[i] = 4;
	}

	// Allocate DST on gpu	
	int* dst;
	cudaMalloc(&dst, SIZE*sizeof(int));

	cudaMemcpyAsync((void*)dst, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice, copyStream);
	
	cudaSetDevice(1);
	int* src_1;
	cudaMalloc(&src_1, SIZE*sizeof(int));

	cudaMemcpy((void*)src_1, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaSetDevice(2);
	int* src_2;
	cudaMalloc(&src_2, SIZE*sizeof(int));

	cudaMemcpy((void*)src_2, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaSetDevice(3);
	int* src_3;
	cudaMalloc(&src_3, SIZE*sizeof(int));

	cudaMemcpy((void*)src_3, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();


	cudaSetDevice(0);

	cudaDeviceSynchronize();
	
	for (int i = 0; i < reps; i++) {
	    cudaMemcpyAsync((void*)src, (void*)dst, sizeof(int)*SIZE , cudaMemcpyDeviceToHost, copyStream);
	    if (nlinks >=1)  
	    	cudaMemcpyAsync((void*)src_1, (void*)dst, sizeof(int)*SIZE , cudaMemcpyDeviceToHost, copyStream2);
	    if (nlinks >= 2)
	    	cudaMemcpyAsync((void*)src_2, (void*)dst, sizeof(int)*SIZE , cudaMemcpyDeviceToHost, copyStream3);
	    if (nlinks >= 3)
	    	cudaMemcpyAsync((void*)src_3, (void*)dst, sizeof(int)*SIZE , cudaMemcpyDeviceToHost, copyStream4);
        }

	// Create a handle for CUBLAS	        
	cudaStreamSynchronize(copyStream);
	cudaStreamSynchronize(copyStream2);
	cudaStreamSynchronize(copyStream3);
	cudaStreamSynchronize(copyStream4);

	//Free pinned memory
	cudaFreeHost(src);
	cudaFree(dst);
	
	cudaSetDevice(1);
	cudaFree(src_1);
	
	cudaSetDevice(2);
	cudaFree(src_2);

	cudaSetDevice(3);
	cudaFree(src_3);


	result = cudaStreamDestroy(copyStream);
	result = cudaStreamDestroy(copyStream2);
	result = cudaStreamDestroy(copyStream3);
	result = cudaStreamDestroy(copyStream4);
	return 0;
}
