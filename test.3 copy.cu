// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <cstdio>
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

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}


void setup_nvlink(gpu){
	int numGPUs = 4;

	int i = gpu;
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

void fill_sin(float *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = sin(float(i + j * nr_rows_A));
}

int main(int argc, char* argv[]) {
        
	if (argc != 8){
		std::cout << "USAGE: " << argv[0] <<"<size> <reps> <active-links>" <<std::endl ;
		exit(-1);
	}

	int multiplier = atoi(argv[1]);

	int reps = atoi(argv[2]);
	
	int nlinks = atoi(argv[3]);

	int gpu = atoi(argv[4]);
	int d1 = atoi(argv[5]);
	int d2 = atoi(argv[6]);
	int d3 = atoi(argv[7]);
        
    setup_nvlink(gpu);

	cudaSetDevice(gpu);
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
		src[i] = sin(i);
		src_h[i] = 4;
	}

	// Allocate DST on gpu	
	int* dst;
	cudaMalloc(&dst, SIZE*sizeof(int));

	cudaMemcpyAsync((void*)dst, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice, copyStream);
	
	cudaSetDevice(d1);
	int* src_1;
	cudaMalloc(&src_1, SIZE*sizeof(int));

	cudaMemcpy((void*)src_1, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaSetDevice(d2);
	int* src_2;
	cudaMalloc(&src_2, SIZE*sizeof(int));

	cudaMemcpy((void*)src_2, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	cudaSetDevice(d3);
	int* src_3;
	cudaMalloc(&src_3, SIZE*sizeof(int));

	cudaMemcpy((void*)src_3, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();


	cudaSetDevice(gpu);

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
	
	cudaSetDevice(d1);
	cudaFree(src_1);
	
	cudaSetDevice(d2);
	cudaFree(src_2);

	cudaSetDevice(d3);
	cudaFree(src_3);


	result = cudaStreamDestroy(copyStream);
	result = cudaStreamDestroy(copyStream2);
	result = cudaStreamDestroy(copyStream3);
	result = cudaStreamDestroy(copyStream4);
	return 0;
}
