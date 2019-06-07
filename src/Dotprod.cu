#include <iostream>
#include <cstdio>
#include "DenseVector.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"

int main() {
    int dim = 1024;
    DenseVector<float> vec1 = DenseVectorFactory::constant<float>(dim, 1);
    DenseVector<float> vec2 = DenseVectorFactory::constant<float>(dim, 2);
    
    gpu::device_ptr<float[]> deviceVec1 = gpu::make_device<float>(vec1.entries, dim);
    gpu::device_ptr<float[]> deviceVec2 = gpu::make_device<float>(vec2.entries, dim);
    gpu::device_ptr<float>  deviceResult = gpu::make_device<float>();
    
    kernel::dotProduct<<<1, dim, dim*sizeof(float)>>>(deviceVec1.get(), deviceVec2.get(), dim, deviceResult.get());
    checkCuda(cudaPeekAtLastError());

    float result = 0;
    gpu::memcpy_to_host<float>(&result, deviceResult.get());
    std::cout << "Result: " << result << "\n";
}

