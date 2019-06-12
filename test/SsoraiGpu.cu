#include "SparseMatrix.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"
#include "gtest/gtest.h"

TEST(SsoraInverse, gpu) {
    int dim = 1024;

    auto mat = SparseMatrix<float>::triDiagonal(dim, -1, 2, -1);

    auto ssorai = mat.ssoraInverse(1);
    
    auto mat_gpu = SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>>::triDiagonal(dim , -1, 2, -1);

    auto ssorai_gpu = mat_gpu.ssoraInverse(1);

    auto entries = gpu::get_from_device<float>(ssorai_gpu.entries(), mat.nonZeroEntries());
    auto cols = gpu::get_from_device<int>(ssorai_gpu.cols(), mat.nonZeroEntries());
    auto rowPtrs = gpu::get_from_device<int>(ssorai_gpu.rowPtrs(), dim + 1);

    for (int i = 0; i < mat.nonZeroEntries(); i++) {
	ASSERT_EQ(entries[i], ssorai.entries()[i]);
    }
    for (int i = 0; i < mat.nonZeroEntries(); i++) {
	ASSERT_EQ(cols[i], ssorai.cols()[i]);
    }
    for (int i = 0; i < dim + 1; i++) {
	ASSERT_EQ(rowPtrs[i], ssorai.rowPtrs()[i]);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}