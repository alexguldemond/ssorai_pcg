#include "DenseVector.hpp"
#include "gtest/gtest.h"

TEST(Vector, gpu) {
    int dim = 1024;
    DeviceVector<float> vec1 = DeviceVector<float>::constant(dim , 1);
    DeviceVector<float> vec2 = DeviceVector<float>::zero(dim);

    DeviceVector<float> vec3 = vec2.plusAx(-1, vec1);
    DeviceVector<float> vec4(vec2);

    vec4 = vec2.plusAx(2, vec1);
    
    std::unique_ptr<float[]> entries1 = gpu::get_from_device<float>(vec1.entries(), dim);
    std::unique_ptr<float[]> entries2 = gpu::get_from_device<float>(vec2.entries(), dim);
    std::unique_ptr<float[]> entries3 = gpu::get_from_device<float>(vec3.entries(), dim);
    std::unique_ptr<float[]> entries4 = gpu::get_from_device<float>(vec4.entries(), dim);

    for (int i = 0; i < dim; i++) {
	ASSERT_EQ(1, entries1[i]);
	ASSERT_EQ(0, entries2[2]);
	ASSERT_EQ(-1, entries3[3]);
	ASSERT_EQ(2, entries4[4]);
    }
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}