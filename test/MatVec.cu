#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"
#include "gtest/gtest.h"

TEST(MatVec, gpu) {
    const int dim = 1024*32;
    DenseVector<float, gpu::CudaDeleter<float[]>> vec = DenseVector<float, gpu::CudaDeleter<float[]>>::constant(dim, 1.f);

    std::vector<float> band(100, 1);
    
    SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>> mat =
	SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>>::bandMatrix(dim, band);
    
    DenseVector<float, gpu::CudaDeleter<float[]>> prod = mat * vec;
    
    float result[dim];
    gpu::memcpy_to_host(result, prod.entries().get(), dim);

    for (int i = 0; i < band.size(); i++) {
	ASSERT_EQ(100 + i, result[i]);
    }
    
    for(int i = band.size(); i < dim - band.size(); i++) {
	ASSERT_EQ(199, result[i]);
    }

    for (int i = dim - band.size(); i < dim; i++) {
	ASSERT_EQ(99 + dim - i, result[i]);
    }
}

TEST(MatVec, cpu) {
    const int dim = 1024;
    DenseVector<float> vec = DenseVector<float>::constant(dim, 1.f);
    SparseMatrix<float> mat = SparseMatrix<float>::triDiagonal(dim, 1, 2, 3);
    
    DenseVector<float> result = mat * vec;
    
    ASSERT_EQ(5, result[0]);
    
    for(int i = 1; i < dim - 1; i++) {
	ASSERT_EQ(6, result[i]);
    }

    ASSERT_EQ(3, result[dim - 1]);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}