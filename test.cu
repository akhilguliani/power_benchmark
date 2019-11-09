/*
 ============================================================================
 Name        : test.cu
 Author      : Akhil Guliani
 Version     :
 Copyright   : Your copyright notice
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <cstdio>
#include <vector>

#include <cublas_v2.h>
#include <curand.h>

using namespace std;

// This kernel is for demonstration purposes only, not a performant kernel for
__global__ void inc_kernel(int *arr, long num_elems) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;

// #pragma unroll(5)
  for (size_t i = globalId; i < num_elems; i += gridSize) {
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
    arr[i] = arr[i] * 2;
  }
}

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
  // Create a pseudo-random number generator
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

  // Set the seed for the random number generator using the system clock
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

  // Fill the array with random numbers on the device
  curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void init_compute_kernel(float* d_A, float* d_B, float* d_C, int dim) {
  // Allocate 3 arrays on CPU
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

  // for simplicity we are going to use square arrays
  nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = dim;

  // Allocate 3 arrays on GPU
  cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
  cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
  cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

  // Fill the arrays A and B on GPU with random numbers
  GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
  GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
}

void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const
                   int k, const int n, cublasHandle_t handle, cudaStream_t stream) {
  int lda=m,ldb=k,ldc=m;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;

  cublasSetStream(handle, stream);

  // Do the actual multiplication
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

  // Destroy the handle
  cublasDestroy(handle);
}

// TODO: write a checker to make sure inc_kernel actually was called
int check_arr(int* arr, int size, int val) {
  for (int i = 0; i < size; ++i) {
    if (arr[i] != val) {
      printf("Got value %d at idx %d\n", arr[i], i);
      return i + 1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  int* dst;
  long size = 1*1024*1024*1024;
  int* src = (int*) malloc(size * sizeof(int));
  float *d_A, *d_B, *d_C;
  int dim = 128;

  for (int i = 0; i < size ; ++i) {
    src[i] = 5;
  }

  cudaMalloc(&dst, size*sizeof(int));

  cudaStream_t copy_stream, compute_stream;
  cudaStreamCreate(&copy_stream);
  cudaStreamCreate(&compute_stream);

    // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  // init_compute_kernel(d_A, d_B, d_C, dim);

  cudaMemcpyAsync((void*)dst, (void*)src, sizeof(int) * size, cudaMemcpyHostToDevice, copy_stream);
  cudaDeviceSynchronize();
  
  for( int i=0; i<100; i++){
  // Just do memcpy on copy_stream;
  // cudaMemcpyAsync((void*)dst, (void*)src, sizeof(int) * size, cudaMemcpyHostToDevice, copy_stream);
  // increment
  inc_kernel<<<32, 32, 0, copy_stream>>>((int*) dst, (long) size);
  // copy back
  // cudaMemcpyAsync((void*)src, (void*)dst, sizeof(int) * size, cudaMemcpyDeviceToHost, copy_stream);
  }
  // gpu_blas_mmul(d_A, d_B, d_C, dim, dim, dim, handle, compute_stream);
  cudaMemcpyAsync((void*)src, (void*)dst, sizeof(int) * size, cudaMemcpyDeviceToHost, copy_stream);

  cudaStreamSynchronize(copy_stream);
  cudaStreamSynchronize(compute_stream);
  // cudaDeviceSynchronize();

  /*
  int idx = check_arr(src, size, 6);
  if (idx) {
    printf("Memcpy test failed at index %d!\n", (idx - 1));
  }*/

  exit(0);
}
