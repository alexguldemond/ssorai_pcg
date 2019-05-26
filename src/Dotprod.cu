#include <iostream>
#include <cstdio>
#include "DenseVector.hpp"
#include "DevicePtr.cuh"

template<class T>
__global__ void dotProd(const T* vector1, const T* vector2, int* dim,T* result) {
    extern __shared__ int shared[];
    T* cache = (T*) shared;
    
    const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
    const int cacheindex = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    T temp = 0;
    for (int i = globalId; i < *dim; i += stride) {
	temp += vector1[i] * vector2[i];
    }
    cache[cacheindex] = temp;
    
    __syncthreads();
    int i = blockDim.x / 2;
    while (i > 0) {
	if( threadIdx.x < i ) {
	    cache[threadIdx.x] += cache[threadIdx.x + i];
	}
	__syncthreads();
	i /= 2;
    }

    if (threadIdx.x == 0) {
	atomicAdd(result, cache[0]);
    }
}

int main() {
    int dim = 999;
    DenseVector<float> vec1 = DenseVector<float>::constant(dim, 1);
    DenseVector<float> vec2 = DenseVector<float>::constant(dim, 2);

    DevicePtr<float> deviceVec1(vec1.data(), dim);
    DevicePtr<float> deviceVec2(vec2.data(), dim);
    DevicePtr<int> deviceDim(&dim);
    DevicePtr<float> deviceResult;

    dotProd<<<4, 256, 256*sizeof(float)>>>(deviceVec1.raw(), deviceVec2.raw(), deviceDim.raw(), deviceResult.raw());
    checkCuda(cudaPeekAtLastError());

    float result = -1.0;
    deviceResult.copyToHost(&result);
    
    std::cout << "Result: " << result << "\n";
}

