#ifndef PARALLEL_CUH
#define PARALLEL_CUH

#define THREADS_PER_VECTOR 32

namespace kernel {

    template<class T>
    __global__ void copyArray(const T* __restrict__ src, T* __restrict__ dest, int dim) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	for (int i = globalId; i < dim; i += stride) {
	    dest[i] = src[i];
	}
    }
    
    template<class T>
    __global__ void aXPlusY(T scalar, const T* __restrict__ vector1, const T* __restrict__ vector2,  int dim , T* __restrict__ result) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	for (int i = globalId; i < dim; i += stride) {
	    result[i] = scalar * vector1[i] + vector2[i];
	}
    }
    
    template<class T>
    __global__ void dotProduct(const T* __restrict__ vector1, const T* __restrict__ vector2, int dim,T* __restrict__ result) {
	extern __shared__ int shared[];
	T* cache = (T*) shared;
	
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int cacheindex = threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	
	T temp = 0;
	for (int i = globalId; i < dim; i += stride) {
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

    template <class T>
    __global__ void sparseMatrixVectorProduct(const T* __restrict__  matEntries,
					      const int* __restrict__ matCols,
					      const int* __restrict__ matRowPtrs,
					      const T* __restrict__ vec,
					      int dim,
					      T* __restrict__ result) {

	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;

	for (int i = globalId; i < dim; i += stride) {
	    T sum = 0;
	    for (int j = matRowPtrs[i]; j < matRowPtrs[i+1]; j++) {
		sum += matEntries[j] * vec[matCols[j]];
	    }
	    result[i] = sum;
	}
    }	
}

#endif