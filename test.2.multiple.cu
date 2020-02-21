// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cublas_v2.h>
#include <curand.h>
#include <cmath>

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

// Randomization helpers 
// adapted from https://github.com/ROCmSoftwarePlatform/rocBLAS/blob/rocm-3.0/clients/include/rocblas_init.hpp#L42

void fill_sin(float *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = sin(float(i + j * nr_rows_A));
}


void fill_cos(float *A, size_t nr_rows_A, size_t nr_cols_A){
    for(size_t i = 0; i < nr_rows_A; ++i)
        for(size_t j = 0; j < nr_cols_A; ++j)
	    A[i + j * nr_rows_A] = cos(float(i + j * nr_rows_A));
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


void setup_nvlink(int numGPUs){
	if (numGPUs >= 4){
		numGPUs = 3;
	}

	for (int i = 0; i <= numGPUs; i++) {
		for (int j = i+1; j <= numGPUs; j++) {
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

	setup_nvlink(nlinks);

	cudaStream_t computeStream;
	cudaStream_t compStrm1, compStrm2, compStrm3;
	cudaError_t result;

	result = cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&compStrm1, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&compStrm2, cudaStreamNonBlocking);
	result = cudaStreamCreateWithFlags(&compStrm3, cudaStreamNonBlocking);

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
		src[i] = sin(i);
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


	// Allocate 3 arrays on CPU
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	// for simplicity we are going to use square arrays
	nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = size;

	float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
	float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
	float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
	
	// Fill the arrays A and B on GPU with random numbers
	fill_sin(h_A, nr_rows_A, nr_cols_A);
	fill_cos(h_B, nr_rows_B, nr_cols_B);
	
	cudaSetDevice(0);

	cudaDeviceSynchronize();
	// Allocate 3 arrays on GPU
	float *d_A, *d_B, *d_C;
	cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));
	// copy data to device
	cudaMemcpyAsync(d_A,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice, computeStream);
	cudaMemcpyAsync(d_B,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice, computeStream);
	cudaMemcpyAsync(d_C,h_A,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice, computeStream);
	cudaDeviceSynchronize();

	cudaSetDevice(1);
	cudaDeviceSynchronize();
	// Allocate 3 arrays on GPU
	float *d_A_1, *d_B_1, *d_C_1;
	cudaMalloc(&d_A_1,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B_1,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C_1,nr_rows_C * nr_cols_C * sizeof(float));
	// copy data to device
	cudaMemcpyAsync(d_A_1,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice, compStrm1);
	cudaMemcpyAsync(d_B_1,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice, compStrm1);
	cudaMemcpyAsync(d_C_1,h_A,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice, compStrm1);
	
	cudaDeviceSynchronize();

	cudaSetDevice(2);
	cudaDeviceSynchronize();
	// Allocate 3 arrays on GPU
	float *d_A_2, *d_B_2, *d_C_2;
	cudaMalloc(&d_A_2,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B_2,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C_2,nr_rows_C * nr_cols_C * sizeof(float));
	// copy data to device
	cudaMemcpyAsync(d_A_2,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice, compStrm2);
	cudaMemcpyAsync(d_B_2,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice, compStrm2);
	cudaMemcpyAsync(d_C_2,h_A,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice, compStrm2);
	
	cudaDeviceSynchronize();

	cudaSetDevice(3);

	cudaDeviceSynchronize();
	// Allocate 3 arrays on GPU
	float *d_A_3, *d_B_3, *d_C_3;
	cudaMalloc(&d_A_3,nr_rows_A * nr_cols_A * sizeof(float));
	cudaMalloc(&d_B_3,nr_rows_B * nr_cols_B * sizeof(float));
	cudaMalloc(&d_C_3,nr_rows_C * nr_cols_C * sizeof(float));
	// copy data to device
	cudaMemcpyAsync(d_A_3,h_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyHostToDevice, compStrm3);
	cudaMemcpyAsync(d_B_3,h_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice, compStrm3);
	cudaMemcpyAsync(d_C_3,h_A,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyHostToDevice, compStrm3);
	
	cudaDeviceSynchronize();

	std::cout << "A =" << std::endl;
	// print_matrix(h_A, nr_rows_A, nr_cols_A);
	std::cout << "B =" << std::endl;
	// print_matrix(h_B, nr_rows_B, nr_cols_B);
	cudaMemcpyAsync((void*)dst, (void*)src, sizeof(int)*SIZE , cudaMemcpyHostToDevice, copyStream);
	
	cudaSetDevice(0);
	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSetStream(handle, computeStream);
	cudaDeviceSynchronize();

	cudaSetDevice(1);
	cublasHandle_t handle1;
	cublasCreate(&handle1);
	cublasSetStream(handle, compStrm1);
	cudaDeviceSynchronize();

	cudaSetDevice(2);
	cublasHandle_t handle2;
	cublasCreate(&handle2);
	cublasSetStream(handle, compStrm2);
	cudaDeviceSynchronize();

	cudaSetDevice(3);
	cublasHandle_t handle3;
	cublasCreate(&handle3);
	cublasSetStream(handle, compStrm3);
	cudaDeviceSynchronize();

	cudaSetDevice(0);

	for (int j = 0 ; j < 100; j++){
		for (int gpu=0; gpu <= nlinks; gpu++){ 
			switch(gpu){
				case 0:
					// Takes about 5 minuets
					cudaSetDevice(0);
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
					break;
				
				case 1:
					cudaSetDevice(1);
					gpu_blas_mmul(handle1, d_A_1, d_B_1, d_C_1, nr_rows_A, nr_cols_A, nr_cols_B);
					break;
				
				case 2:
					cudaSetDevice(2);
					gpu_blas_mmul(handle2, d_A_2, d_B_2, d_C_2, nr_rows_A, nr_cols_A, nr_cols_B);
					break;
				case 3:
					cudaSetDevice(3);
					gpu_blas_mmul(handle3, d_A_3, d_B_3, d_C_3, nr_rows_A, nr_cols_A, nr_cols_B);
					break;
			}
			// Create a handle for CUBLAS
		}
		cudaSetDevice(0);
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
	
	cudaSetDevice(1);
	cudaStreamSynchronize(compStrm1);
	cudaSetDevice(2);
	cudaStreamSynchronize(compStrm2);
	cudaSetDevice(3);
	cudaStreamSynchronize(compStrm3);  
	
	cudaSetDevice(0);
	cudaDeviceSynchronize();
	// Destroy the handle
	cublasDestroy(handle);


	// Copy (and print) the result on host memory
	cudaMemcpyAsync(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost, computeStream);
	std::cout << "C =" << std::endl;
	// print_matrix(h_C, nr_rows_C, nr_cols_C);

	// cudaMemcpyAsync(h_C,d_C_1,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost, compStrm1);
	// std::cout << "C =" << std::endl;

	// cudaMemcpyAsync(h_C,d_C_2,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost, compStrm1);
	// std::cout << "C =" << std::endl;

	// cudaMemcpyAsync(h_C,d_C_3,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost, compStrm1);
	// std::cout << "C =" << std::endl;

	//Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	cudaFreeHost(src);
	cudaFreeHost(dest_h);
	cudaFree(dst);	

	cudaSetDevice(1);
	cudaFree(src_1);
	cudaFree(d_A_1);
	cudaFree(d_B_1);
	cudaFree(d_C_1);
	cublasDestroy(handle1);
	result = cudaStreamDestroy(compStrm1);



	cudaSetDevice(2);
	cudaFree(src_2);
	cudaFree(d_A_2);
	cudaFree(d_B_2);
	cudaFree(d_C_2);
	cublasDestroy(handle2);
	result = cudaStreamDestroy(compStrm2);


	cudaSetDevice(3);
	cudaFree(src_3);
	cudaFree(d_A_3);
	cudaFree(d_B_3);
	cudaFree(d_C_3);
	cublasDestroy(handle3);
	result = cudaStreamDestroy(compStrm3);

	cudaSetDevice(0);
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
