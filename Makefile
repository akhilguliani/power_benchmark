all: test test1 test2 testBi testMul test3dev

testMul: test.2.multiple.cu
	nvcc -o $@ $< -lcurand -lcudart -lcublas

testBi: test.bi.cu
	nvcc -o $@ $< -lcurand -lcudart -lcublas

test2: test.2.cu
	nvcc -o $@ $< -lcurand -lcudart -lcublas

test1: test.1.cu
	nvcc -o test1 test.1.cu -lcurand -lcudart -lcublas

test1dev: test.1.dev.cu
	nvcc -o $@ $< -lcurand -lcudart -lcublas

test3dev: test.3.copy.cu
	nvcc -o $@ $< -lcurand -lcudart -lcublas

test: test.cu
	nvcc -o test test.cu -lcurand -lcudart -lcublas

clean:
	rm test test1 test2 testBi testMul
