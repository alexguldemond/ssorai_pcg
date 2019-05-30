#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "DevicePtr.cuh"
#include "Parallel.cuh"

int main () {
    const int dim = 2123;
    DenseVector<float> vec = DenseVector<float>::constant(dim, 1);
    SparseMatrix<float> mat = SparseMatrix<float>::triDiagonal(dim, 1, 2, 3);
    
    DevicePtr<float> deviceEntries(&mat.entries[0], mat.nonZeroEntries());
    DevicePtr<int> deviceCols(&mat.cols[0], mat.nonZeroEntries());
    DevicePtr<int> deviceRowPtrs(&mat.rowPtrs[0], dim + 1);

    DevicePtr<float> deviceVec(vec.data(), dim);

    
    DevicePtr<float> deviceResult(dim);

    int blockSize = (mat.nonZeroEntries()/1024) + 1;
    kernel::sparseMatrixVectorProduct<<<blockSize,
	1024,
	mat.nonZeroEntries() *sizeof(float)>>>(deviceEntries.raw(),
					       deviceCols.raw(),
					       deviceRowPtrs.raw(),
					       deviceVec.raw(),
					       dim,
					       deviceResult.raw());
    checkCuda(cudaPeekAtLastError());
    
    float result[dim];
    deviceResult.copyToHost(result);
    std::cout << "NNZ: " << mat.nonZeroEntries() << "\n";

    if (result[0] != 5) {
	std::cout << "result[0] is not 5: " << result[0] << "\n";
	return 1;
    }
    for(int i = 1; i < dim - 1; i++) {
	if (result[i] != 6) {
	    std::cout << "result[" << i << "] is not 6: " << result[i] << "\n";
	    //return 1;
	}
    }
    if (result[dim - 1] != 3) {
	std::cout << "result[last] is not 3: " << result[dim - 1] << "\n";
	return 1;
    }

    return 0;
}