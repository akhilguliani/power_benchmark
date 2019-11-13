all: test test1

test1: test.1.cu
	nvcc -o test1 test.1.cu -lcurand -lcudart -lcublas

test: test.cu
	nvcc -o test test.cu -lcurand -lcudart -lcublas
