#include <iostream>
#include <cstdio>
#include "DenseVector.hpp"
#include "DevicePtr.cuh"
#include "Parallel.cuh"

int main() {
    int dim = 1024;
    DenseVector<float> vec1 = DenseVector<float>::constant(dim, 1);
    DenseVector<float> vec2 = DenseVector<float>::constant(dim, 2);

    DevicePtr<float> deviceVec1(vec1.data(), dim);
    DevicePtr<float> deviceVec2(vec2.data(), dim);
    DevicePtr<float> deviceResult;

    kernel::dotProduct<<<1, dim, dim*sizeof(float)>>>(deviceVec1.raw(), deviceVec2.raw(), dim, deviceResult.raw());
    checkCuda(cudaPeekAtLastError());

    float result = deviceResult.get();
    
    std::cout << "Result: " << result << "\n";
}

