#include <iostream>
#include <cstdio>
#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "DevicePtr.cuh"

template <class T>
__global__ void sparseMatrixVectorProd(const T* matEntries,
				       const int* matCols,
				       const int* matRowPtrs,
				       const T* vec,
				       const int* dim,
				       T* result) {
    extern __shared__ int shared[];
    T* cache = (T*) shared;

    const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int nnz = matRowPtrs[*dim];
    
    for (int i = globalId; i < nnz; i += stride) {
	cache[i] = matEntries[i] * vec[matCols[i]];
    }
    __syncthreads();
    
    for (int i = 0; i < *dim; i++) {
	if (globalId == matRowPtrs[i]) {
	    result[i] = 0;
	    for (int j = matRowPtrs[i]; j < matRowPtrs[i+1]; j++) {
		atomicAdd(&result[i], cache[j]);
	    }
	}
    }
}

int main () {
    const int dim = 32;
    DenseVector<float> vec = DenseVector<float>::constant(dim, 1);
    SparseMatrix<float> mat = SparseMatrix<float>::triDiagonal(dim, 1, 2, 3);
    
    DevicePtr<float> deviceEntries(&mat.entries[0], mat.nonZeroEntries());
    DevicePtr<int> deviceCols(&mat.cols[0], mat.nonZeroEntries());
    DevicePtr<int> deviceRowPtrs(&mat.rowPtrs[0], dim + 1);

    DevicePtr<float> deviceVec(vec.data(), dim);

    DevicePtr<int> deviceDim(&dim);
    
    DevicePtr<float> deviceResult(dim);
    
    sparseMatrixVectorProd<<<2,mat.nonZeroEntries() ,mat.nonZeroEntries() *sizeof(float)>>>(deviceEntries.raw(),
											    deviceCols.raw(),
											    deviceRowPtrs.raw(),
											    deviceVec.raw(),
											    deviceDim.raw(),
											    deviceResult.raw());
    checkCuda(cudaPeekAtLastError());

    float result[dim];
    deviceResult.copyToHost(result);

    for(int i = 0; i < dim; i++) {
	std::cout << result[i] << "\n";
    }


    return 0;
}