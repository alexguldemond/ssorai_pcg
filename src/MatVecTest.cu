#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "DevicePtr.cuh"
#include "Parallel.cuh"
#include "GpuSolve.cuh"

int main() {
    const int dim = 32;
    const int threadsPerBlock = 1024;

    std::vector<float> zVec{-2.68149,2.79199,8.5675,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,10.0114,7.84556,2.43102,0.176801};

    DenseVector<float> z(zVec);

    SparseMatrix<float> a = SparseMatrix<float>::triDiagonal(dim, -1, 2, -1);

    DeviceMatrix<float> device_a(a);

    DevicePtr<float> deviceVec(z.data(), dim);

    DevicePtr<float> deviceResult(dim);

    int nonZero = a.nonZeroEntries();
    int gridSize = (nonZero/threadsPerBlock) + (nonZero % threadsPerBlock != 0);
    int sharedSize = nonZero * sizeof(float);
    kernel::sparseMatrixVectorProduct<<<gridSize,
	threadsPerBlock,
	sharedSize>>>(device_a.entries.raw(),
		      device_a.cols.raw(),
		      device_a.rowPtrs.raw(),
		      deviceVec.raw(),
		      dim,
		      deviceResult.raw());
    checkCuda(cudaPeekAtLastError());

    float gpuResult[dim];
    deviceResult.copyToHost(gpuResult);
    
    DenseVector<float> cpuResult = a * z;

    for (int i = 0; i < dim; i++) {
	
	std::cout << "gpuResult[ " << i << " ] = " << gpuResult[i] << ", cpuResult[ " << i << " ] = " << cpuResult[i] << "\n";
    }
    
}