all: test test1 test2 testBi

testBi: test.bi.cu
	nvcc -o $@ $< -lcurand -lcudart -lcublas

test2: test.2.cu
	nvcc -o $@ $< -lcurand -lcudart -lcublas

test1: test.1.cu
	nvcc -o test1 test.1.cu -lcurand -lcudart -lcublas

test: test.cu
	nvcc -o test test.cu -lcurand -lcudart -lcublas
