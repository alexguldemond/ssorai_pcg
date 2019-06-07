#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"

int main () {
    const int dim = 1024;
    DenseVector<float> vec = DenseVectorFactory::constant<float>(dim, 1);
    SparseMatrix<float> mat = SparseMatrixFactory::triDiagonal<float>(dim, 1, 2, 3);
    const int nnz = mat.nonZeroEntries();
    
    gpu::device_ptr<float[]> deviceEntries(gpu::make_device<float>(mat.entries, nnz));
    gpu::device_ptr<int[]> deviceCols(gpu::make_device<int>(mat.cols, nnz));
    gpu::device_ptr<int[]> deviceRowPtrs(gpu::make_device<int>(mat.rowPtrs, dim + 1));

    gpu::device_ptr<float[]> deviceVec(gpu::make_device<float>(vec.entries, dim));
    
    gpu::device_ptr<float[]> deviceResult(gpu::make_device<float>(dim)); 

    std::cout << "dim = " << dim << "\n";
    std::cout << "NNZ = " << mat.nonZeroEntries() << "\n";
    

    kernel::sparseMatrixVectorProduct<<<dim / 1024,1024>>>(deviceEntries.get(),
							   deviceCols.get(),
							   deviceRowPtrs.get(),
							   deviceVec.get(),
							   dim,
							   deviceResult.get());
    checkCuda(cudaPeekAtLastError());
    
    float result[dim];
    gpu::memcpy_to_host(result, deviceResult.get(), dim);
    
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
    std::cout << "Dont checking\n";
    return 0;
    
}