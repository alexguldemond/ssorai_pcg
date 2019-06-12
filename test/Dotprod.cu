#include "DenseVector.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"
#include "gtest/gtest.h"
TEST(DotProd, dotProd) {
    int dim = 1024;

    DeviceVector<float> vec1 = DenseVector<float, gpu::CudaDeleter<float[]>>::constant(dim, 1);
    DeviceVector<float> vec2 = DenseVector<float, gpu::CudaDeleter<float[]>>::incremental(dim, 1);

    ASSERT_EQ( dim * (dim + 1)/2, vec1.dot(vec2));
    
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}