#ifndef PARALLEL_CUH
#define PARALLEL_CUH

namespace kernel {

    template<class T>
    __global__ void copyArray(const T* src, T* dest, int dim) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	for (int i = globalId; i < dim; i += stride) {
	    dest[i] = src[i];
	}
    }
    
    template<class T>
    __global__ void aXPlusY(T scalar, const T* vector1, const T* vector2,  int dim , T* result) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	for (int i = globalId; i < dim; i += stride) {
	    result[i] = scalar * vector1[i] + vector2[i];
	}
    }
    
    template<class T>
    __global__ void dotProduct(const T* vector1, const T* vector2, int dim,T* result) {
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
	if (globalId == 0) {
	    *result = 0;
	}
	
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
    __global__ void sparseMatrixVectorProduct(const T* matEntries,
					      const int* matCols,
					      const int* matRowPtrs,
					      const T* vec,
					      int dim,
					      T* result) {
	extern __shared__ int shared[];
	T* cache = (T*) shared; //Assume size is equal to nnz
	
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	const int nnz = matRowPtrs[dim];
	
	for (int i = globalId; i < nnz; i += stride) {
	    cache[i] = matEntries[i] * vec[matCols[i]];
	}
	for (int i = globalId; i < dim; i += stride) {
	    result[i] = 0;
	}
	
	__syncthreads();
	
	__syncthreads();
	for (int i = 0; i < dim; i++) {
	    if (globalId == matRowPtrs[i] || ( threadIdx.x == 0 && globalId >= matRowPtrs[i] && globalId < matRowPtrs[i+1]) ) {
		for (int j = globalId + 1; j < matRowPtrs[i + 1]; j++) {
		    cache[globalId] += cache[j];
		}
	    atomicAdd(&result[i], cache[globalId]);
	    }
	}
    }
    
}
#endif