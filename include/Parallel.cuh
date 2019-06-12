#ifndef PARALLEL_CUH
#define PARALLEL_CUH

#define THREADS_PER_VECTOR 32

#include <cstdio>

namespace kernel {
    int roundUpDiv(int dim, int threadsPerBlock) {
	return dim / threadsPerBlock + (((dim % threadsPerBlock) == 0) ? 0 : 1);
    }
    
    template<class T>
    __global__ void initConstArray( T* __restrict__ arr, int dim, T value) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	for (int i = globalId; i < dim; i += stride) {
	    arr[i] = value;
	}
    }

    template<class T>
    __global__ void initIncrementalArray( T* __restrict__ arr, int dim, T startValue) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	for (int i = globalId; i < dim; i += stride) {
	    arr[i] = startValue + i;
	}
    }
    
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

    
    template <class T, unsigned int warpDim>
    __global__ void sparseMatrixVectorProduct(const T* __restrict__  matEntries,
					      const int* __restrict__ matCols,
					      const int* __restrict__ matRowPtrs,
					      const T* __restrict__ vec,
					      int dim,
					      T* __restrict__ result) {
	extern __shared__ volatile int mem[];
	volatile T* sData = (T*) mem;
	volatile int* ptrs = &mem[blockDim.x + warpDim/2];
	
	const int threadId   = blockDim.x * blockIdx.x + threadIdx.x;  // global thread index
	const int threadLane = threadIdx.x & (warpDim - 1);            // thread index within the warp, assumes warpDim is a multiple of 2 to perform fast modulo operation
	const int warpId   = threadId  /  warpDim;                     // global warp index
	const int warpLane = threadIdx.x /  warpDim;                   // warp index within the block
	const int warpsPerBlock = blockDim.x / warpDim;
	const int numWarps = warpsPerBlock * gridDim.x;                // total number of active vect

	for (int row = warpId; row < dim; row += numWarps) {

	    //First two threads cache ptrs
	    if (threadLane < 2) {
		ptrs[warpLane * 2 + threadLane] = matRowPtrs[row + threadLane];	
	    }

	    const int rowStart = ptrs[warpLane * 2];
	    const int rowEnd   = ptrs[warpLane * 2 + 1];
	    T sum = 0;
	    if (rowEnd - rowStart > warpDim && warpDim == 32) {

		int j = rowStart - (rowStart & (warpDim - 1)) + threadLane;
		if (j >= rowStart && j < rowEnd) {
		    sum += matEntries[j] * vec[matCols[j]];
		}

		for (j = j + warpDim; j < rowEnd; j += warpDim) {
		    sum += matEntries[j] * vec[matCols[j]];
		}
	    } else {
		for (int j = rowStart + threadLane; j < rowEnd; j += warpDim) {
		    sum += matEntries[j] * vec[matCols[j]];
		}
	    }

	    sData[threadIdx.x] = sum;

	    // reduce local sums to row sum
	    if (warpDim > 16) {
		sData[threadIdx.x] = sum = sum + sData[threadIdx.x + 16];
	    }
	    if (warpDim >  8) {
		sData[threadIdx.x] = sum = sum + sData[threadIdx.x +  8];
	    }
	    if (warpDim >  4) {
		sData[threadIdx.x] = sum = sum + sData[threadIdx.x +  4];
	    }
	    if (warpDim >  2) {
		sData[threadIdx.x] = sum = sum + sData[threadIdx.x +  2];
	    }
	    if (warpDim >  1) {
		sData[threadIdx.x] = sum = sum + sData[threadIdx.x +  1];
	    }

	    if (threadLane == 0) {
		result[row] = sData[threadIdx.x];
	    }
	}
    }

    template <class T>
    __device__ T getDiagonal(const T* __restrict__  entries,
			     const int* __restrict__ cols,
			     const int* __restrict__ rowPtrs,
			     int dim,
			     int i) {
	int start = rowPtrs[i];
	int end = rowPtrs[i+1];
	int middle;
	while (start <= end) {
	    middle = start + (end - start) / 2;
	    if (cols[middle] == i) {
		return entries[middle];
	    }
	    if (cols[middle] < i) {
		start = middle + 1;
	    } else {
		end = middle	- 1;
	    }
	}
	return 0;
    }
    
    template <class T>
    __global__ void getAllDiagonals(const T* __restrict__  matEntries,
				    const int* __restrict__ matCols,
				    const int* __restrict__ matRowPtrs,
				    int dim,
				    T* __restrict__ result) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	for (int i = globalId; i < dim; i += stride) {
	    result[i] = getDiagonal<T>(matEntries, matCols, matRowPtrs, dim,i);
	}
    }

    template <class T>
    __global__ void ssoraiEntries(const T* __restrict__ entries,
				  const int* __restrict__ cols,
				  const int* __restrict__ rowPtrs,
				  const T* __restrict__ diagonals,
				  int dim,
				  T relax,
				  T* __restrict__ result) {
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	const int nnz = rowPtrs[dim];
	for (int i = globalId; i < dim; i += stride) {
	    for (int ptr = rowPtrs[i]; ptr < rowPtrs[i+1]; ptr++) {
		int j = cols[ptr];
		
		//Handling symmetry
		int row;
		int col;
		if (i <= j) {
		    row = i;
		    col = j;
		} else {
		    row = j;
		    col = i;
		}
		
		T tmp = 0;
		
		int iPtr = rowPtrs[row];
		int jPtr = rowPtrs[col];
		
		while (iPtr < rowPtrs[row + 1] && jPtr < rowPtrs[col + 1] && cols[jPtr] < row) {
		    if (jPtr >= nnz || iPtr >= nnz) {
			continue;
		    }
		    
		    if (cols[jPtr] == cols[iPtr]) {
			tmp += entries[jPtr] * entries[iPtr] / diagonals[cols[jPtr]];
			jPtr++;
			iPtr++;
		    } else if (cols[jPtr] < cols[iPtr]) {
			jPtr++;
		    } else {
			iPtr++;
		    }
		}
		
		tmp *= relax*relax / (diagonals[row] * diagonals[col]);
		tmp += (i == j) ? 1.0/diagonals[row] : -1.0 * relax * entries[ptr] / (diagonals[row] * diagonals[col]);
		tmp *= relax*(2 - relax);
		result[ptr] = tmp;
	    }
	}
    }

    template<class T>
    __global__ void triDiagonal(int dim, T left, T middle, T right, T* entries, int* cols, int *rowPtrs) {
	
	const int globalId = blockDim.x * blockIdx.x + threadIdx.x;
	const int stride = blockDim.x * gridDim.x;
	const int nonZero = 3 * dim - 2;
	
	if (globalId == 0) {
	    entries[0] = middle;
	    entries[1] = right;
	    cols[0] = 0;
	    cols[1] = 1;
	    rowPtrs[0] = 0;
	    rowPtrs[1] = 2;

	    entries[nonZero - 2] = left;
	    entries[nonZero - 1] = middle;
	    cols[nonZero - 2] = dim - 2;
	    cols[nonZero - 1] = dim -1;
	    rowPtrs[dim] = nonZero;
	}
	for (int i = globalId; i < dim - 2; i += stride) {
	    entries[2 + 3*i] = left;
	    entries[3 + 3*i] = middle;
	    entries[4 + 3*i] = right;
	    cols[2 + 3*i] = i;
	    cols[3 + 3*i] = i + 1;
	    cols[4 + 3*i] = i + 2;
	    rowPtrs[i+2] = 2 + 3*(i + 1);
	}
    }
    
}

#endif