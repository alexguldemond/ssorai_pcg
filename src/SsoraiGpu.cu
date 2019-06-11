#include <iostream>
#include <cstdio>
#include "SparseMatrix.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"

int main() {
    int dim = 1024;

    SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>> mat = SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>>::triDiagonal(dim , -1, 2, -1);

    std::cout << mat.ssoraInverse(1).toString() << "\n";
}
