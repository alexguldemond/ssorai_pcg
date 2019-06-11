#include <iostream>
#include <cstdio>
#include "DenseVector.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"

int main() {
    int dim = 1024;

    std::cout << "Making vec 1\n";
    DeviceVector<float> vec1 = DenseVector<float, gpu::CudaDeleter<float[]>>::constant(dim, 1);

    std::cout << "Making vec 2\n";
    DeviceVector<float> vec2 = DenseVector<float, gpu::CudaDeleter<float[]>>::constant(dim, 2);
    
    std::cout << "Result: " << vec1.dot(vec2) << "\n";
}

