// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
	// Create a pseudo-random number generator
	curandGenerator_t prng;
	curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

	// Set the seed for the random number generator using the system clock
	curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

	// Fill the array with random numbers on the device
	curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}


#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
	switch (error)
	{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";

		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";

		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";

		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";

		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";

		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";

		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";

		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}
#endif

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


// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul( cublasHandle_t handle, const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Do the actual multiplication
	cublasStatus_t err = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	if (err != CUBLAS_STATUS_SUCCESS)
		std::cout << "Error: " <<  _cudaGetErrorEnum(err) << std::endl;

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

int main(int argc, char* argv[]) {

	if (argc != 4){
		std::cout << "USAGE: " << argv[0] <<" <size> <inner-reps> nlinks" <<std::endl ;
		exit(-1);
	}
	int size = atoi(argv[1]);
	int reps = atoi(argv[2]);
	int nlinks = atoi(argv[3]);

	setup_nvlink();

	cudaStream_t computeStream;
	cudaError_t result;

	result = cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking);

	cudaStream_t copyStream, copyStream2;
	cudaStream_t copyStream3, copyStream4;
	result = cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&copyStream2, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&copyStream3, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&copyStream4, cudaStreamNonBlocking);



	// Allocate the src on CPU
	long SIZE = 512*1024*1024;
	// int* src = (int*) malloc(SIZE * sizeof(int));
	int* src; 
	int *dest_h; 
	cudaMallocHost((void**) &src, SIZE * sizeof(int));
	cudaMallocHost((void**) &dest_h, SIZE * sizeof(int));
	for (int i = 0; i < SIZE ; ++i) {
		src[i] = 5;
		dest_h[i] = 1;
	}

	cudaSetDevice(0);
	// Allocate DST on gpu	
	int* dst;
	cudaMalloc(&dst, SIZE*sizeof(int));

	// Allocate buffers on all cpus:
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



	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = size;

	float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

	// If you already have useful values in A and B you can copy them in GPU:
	// cudaMemcpy(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice);
	// cudaMemcpy(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice);

	// Fill the arrays A and B on GPU with random numbers
	GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
	GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);

	// Optionally we can copy the data back on CPU and print the arrays
	cudaMemcpyAsync(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost, computeStream);
	cudaMemcpyAsync(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost, computeStream);
	std::cout << "A =" << std::endl;
	// print_matrix(h_A, nr_rows_A, nr_cols_A);
	std::cout << "B =" << std::endl;
	// print_matrix(h_B, nr_rows_B, nr_cols_B);
	cudaMemcpyAsync((void*)dst, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice, copyStream);

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetStream(handle, computeStream);

	cudaDeviceSynchronize();

	for (int j = 0 ; j < 100; j++){
		// Tabkes about 5 minuets
		gpu_blas_mmul(handle, d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
		for (int i=0; i< reps; i++){
			// each stable copy takes about 162 miliseconds
			// cudaMemcpyAsync((void*)src, (void*)dst, sizeof(int) * SIZE, cudaMemcpyDeviceToHost, copyStream);
			// cudaMemcpyAsync((void*)dest_h, (void*)dst, sizeof(int) * SIZE, cudaMemcpyDeviceToHost, copyStream2);
			if (nlinks >=1)  
				cudaMemcpyAsync((void*)src_1, (void*)dst, sizeof(int)*SIZE , cudaMemcpyDeviceToDevice, copyStream2);
			if (nlinks >= 2)
				cudaMemcpyAsync((void*)src_2, (void*)dst, sizeof(int)*SIZE , cudaMemcpyDeviceToDevice, copyStream3);
			if (nlinks >= 3)
				cudaMemcpyAsync((void*)src_3, (void*)dst, sizeof(int)*SIZE , cudaMemcpyDeviceToDevice, copyStream4);

		}
		// Create a handle for CUBLAS	        
		cudaStreamSynchronize(copyStream);
		cudaStreamSynchronize(copyStream2);
		cudaStreamSynchronize(copyStream3);
		cudaStreamSynchronize(copyStream4);

	}
	cudaStreamSynchronize(computeStream);  
	cudaStreamSynchronize(copyStream);
	cudaStreamSynchronize(copyStream2);
	cudaStreamSynchronize(copyStream3);
	cudaStreamSynchronize(copyStream4);

	// Destroy the handle
	cublasDestroy(handle);

	// Copy (and print) the result on host memory
	cudaMemcpyAsync(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost, computeStream);
	std::cout << "C =" << std::endl;
	// print_matrix(h_C, nr_rows_C, nr_cols_C);

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaFreeHost(src);
	cudaFreeHost(dest_h);
	cudaFree(dst);	

	cudaSetDevice(1);
	cudaFree(src_1);

	cudaSetDevice(2);
	cudaFree(src_2);

	cudaSetDevice(3);
	cudaFree(src_3);


	result = cudaStreamDestroy(computeStream);
	result = cudaStreamDestroy(copyStream);
	result = cudaStreamDestroy(copyStream2);
	result = cudaStreamDestroy(copyStream3);
	result = cudaStreamDestroy(copyStream4);

	// Free CPU memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}
