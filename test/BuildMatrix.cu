#include "SparseMatrix.hpp"
#include "gpu_memory.cuh"
#include "gtest/gtest.h"

TEST(MatrixBuilder, TriDiagonalGPU) {
    int dim = 1024;
    auto mat = SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>>::triDiagonal(dim, -1, 2, -1);
    int nnz = mat.nonZeroEntries();
    ASSERT_EQ(1024*3 - 2, nnz);
    
    std::unique_ptr<float[]> entries = gpu::get_from_device<float>(mat.entries(), mat.nonZeroEntries());
    std::unique_ptr<int[]> cols = gpu::get_from_device<int>(mat.cols(), mat.nonZeroEntries());
    std::unique_ptr<int[]> rowPtrs = gpu::get_from_device<int>(mat.rowPtrs(), dim + 1);

    ASSERT_EQ(2, entries[0]);
    ASSERT_EQ(-1, entries[1]);

    for (int i = 2; i < nnz - 2; i++) {
	if ( (i-2) % 3 == 1) {
	    ASSERT_EQ(2, entries[i]);
	} else {
	    ASSERT_EQ(-1, entries[i]);
	}
    }

    ASSERT_EQ(-1, entries[nnz - 2]);
    ASSERT_EQ(2, entries[nnz - 1]);

    auto hostMat = SparseMatrix<float>::triDiagonal(dim, -1, 2, -1);

    for (int i = 0; i < nnz; i++) {
	ASSERT_EQ(hostMat.cols()[i], cols[i]);
    }

    for (int i = 0; i < dim + 1; i++) {
	ASSERT_EQ(hostMat.rowPtrs()[i], rowPtrs[i]);
    }
}

TEST(MatrixBuilder, TriDiagonalCPU) {
    int dim = 1024;
    auto mat = SparseMatrix<float>::triDiagonal(dim, -1, 2, -1);
    int nnz = mat.nonZeroEntries();
    ASSERT_EQ(1024*3 - 2, nnz);
    ASSERT_EQ(2, mat.entries()[0]);
    ASSERT_EQ(-1, mat.entries()[1]);

    for (int i = 2; i < nnz - 2; i++) {
	if ( (i-2) % 3 == 1) {
	    ASSERT_EQ(2, mat.entries()[i]);
	} else {
	    ASSERT_EQ(-1, mat.entries()[i]);
	}
    }

    ASSERT_EQ(-1, mat.entries()[nnz - 2]);
    ASSERT_EQ(2, mat.entries()[nnz - 1]);
    
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}