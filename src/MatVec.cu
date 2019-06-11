#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"

int main () {
    const int dim = 1024*32;
    DenseVector<float, gpu::CudaDeleter<float[]>> vec = DenseVector<float, gpu::CudaDeleter<float[]>>::constant(dim, 1.f);
    SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>> mat =
	SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>>::triDiagonal(dim, 1, 2, 3);
    
    DenseVector<float, gpu::CudaDeleter<float[]>> prod = mat * vec;
    
    float result[dim];
    gpu::memcpy_to_host(result, prod.entries().get(), dim);
    
    std::cout << "Done, now checking\n";
    if (result[0] != 5) {
	std::cout << "result[0] is not 5: " << result[0] << "\n";
    }
    
    for(int i = 1; i < dim - 1; i++) {
	if (result[i] != 6) {
	    std::cout << "result[" << i << "] is not 6: " << result[i] << "\n";
	}
    }
    if (result[dim - 1] != 3) {
	std::cout << "result[last] is not 3: " << result[dim - 1] << "\n";
    }
    
    std::cout << "Done checking\n\n\n Larger Example:\n";

    const int bandSize = 2000;
    std::vector<float> band(bandSize, 1);
    mat = SparseMatrix<float, gpu::CudaDeleter<float[]>,gpu::CudaDeleter<int[]>>::bandMatrix(dim, band);
    
    prod = mat * vec;
    
    gpu::memcpy_to_host(result, prod.entries().get(), dim);

    std::cout << "Done, now checking\n";
    for (int i = 0; i < bandSize - 1; i++) {
	float target = bandSize + i;
	if (result[i] != target) {
	    std::cout << "result[" << i << "] is not " << target << ": "<< result[i] << "\n";
	}
    }
    for (int i = bandSize - 1; i < dim - bandSize + 1; i++) {;
	if (result[i] != bandSize*2 - 1) {
	    std::cout << "result[" << i << "] is not " << 39 << ": "<< result[i] << "\n";
	}
    }
    for (int i = dim - bandSize + 1; i < dim; i++) {
	float target = bandSize - 1 - i + dim;
	if (result[i] != target) {
	    std::cout << "result[" << i << "] is not " << target << ": "<< result[i] << "\n";
	}
    }
   
    
    return 0;
    
    
}