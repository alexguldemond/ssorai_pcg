#ifndef SPARSEMATRIXBASE_HPP
#define SPARSEMATRIXBASE_HPP

#include "DenseVector.hpp"
#include <string>
#include <sstream>
#include <algorithm>
#include <utility>
#include <vector>

template <class T, class Deleter, class IntDeleter >
class ProxySparseMatrix;

template <class T, class Deleter, class IntDeleter>
class SparseMatrixBase {
private:
    int _dim;
    
public:
    SparseMatrixBase(int dim) : _dim(dim) { }
    
    virtual const std::unique_ptr<T[], Deleter>& entries() const = 0;

    virtual const std::unique_ptr<int[], IntDeleter>& cols() const = 0;

    virtual const std::unique_ptr<int[], IntDeleter>& rowPtrs() const = 0;
    
    int dim() const {
	return _dim;
    }

    int nonZeroEntries() const {
	return rowPtrs()[_dim];
    }

    T operator()(int row, int col) const;
    
    T getDiagonal(int i) const;

    std::unique_ptr<T[], Deleter> getAllDiagonals() const;

    std::unique_ptr<T[], Deleter> ssoraInverseEntries(double w) const;
    
    ProxySparseMatrix<T, Deleter, IntDeleter> ssoraInverse(double w) const;
    
    std::string toString() const;

    void matVec(const DenseVector<T, Deleter>& vec, DenseVector<T, Deleter>& result) const;
    
    MatVec<T, Deleter, IntDeleter> operator*(const DenseVector<T, Deleter>& vec) const;
};

template <class T, class Deleter, class IntDeleter>
T SparseMatrixBase<T, Deleter, IntDeleter>::operator()(int row, int col) const {
    for (int j = rowPtrs()[row]; j < rowPtrs()[row+1]; j++) {
	if (cols()[j] == col) {
	    return entries()[j];
	}
    }
    return 0;
}

template <class T, class Deleter, class IntDeleter>
T SparseMatrixBase<T, Deleter, IntDeleter>::getDiagonal(int i) const {
    int start = rowPtrs()[i];
    int end = rowPtrs()[i+1];
    int middle;
    while (start <= end) {
	middle = start + (end - start) / 2;
	if (cols()[middle] == i) {
	    return entries()[middle];
	}
	if (cols()[middle] < i) {
	    start = middle + 1;
	} else {
	    end = middle - 1;
	}
    }
    return 0;
}

template <class T, class Deleter, class IntDeleter>
std::unique_ptr<T[], Deleter> SparseMatrixBase<T, Deleter, IntDeleter>::getAllDiagonals() const {
    T* diagonals = Allocater<T, Deleter>::allocate(dim());
    for (int i = 0; i < dim(); i++) {
	diagonals[i] = getDiagonal(i);
    }
    return std::unique_ptr<T[], Deleter>(diagonals, Deleter());
}

template <class T, class Deleter, class IntDeleter>
std::unique_ptr<T[], Deleter> SparseMatrixBase<T, Deleter, IntDeleter>::ssoraInverseEntries(double w) const {
    double mult = w*(2 - w);
    int nnz = nonZeroEntries();
    T* newEntries = Allocater<T, Deleter>::allocate(nnz);
    std::unique_ptr<T[], Deleter>  diagonals = getAllDiagonals();
    for (int i = 0; i < dim(); i++) {
	for (int ptr = rowPtrs()[i]; ptr < rowPtrs()[i+1]; ptr++) {
	    int j = cols()[ptr];

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
	    
	    int iPtr = rowPtrs()[row];
	    int jPtr = rowPtrs()[col];

	    while (iPtr < rowPtrs()[row + 1] && jPtr < rowPtrs()[col + 1] && cols()[jPtr] < row) {
		if (jPtr >= nnz || iPtr >= nnz) {
		    continue;
		}
		
		if (cols()[jPtr] == cols()[iPtr]) {
		    tmp += entries()[jPtr] * entries()[iPtr] / diagonals[cols()[jPtr]];
		    jPtr++;
		    iPtr++;
		} else if (cols()[jPtr] < cols()[iPtr]) {
		    jPtr++;
		} else {
		    iPtr++;
		}
	    }
	    
	    tmp *= w*w / (diagonals[row] * diagonals[col]);
	    tmp += (i == j) ? 1.0/diagonals[row] : -1.0 * w * entries()[ptr] / (diagonals[row] * diagonals[col]);
	    tmp *= mult;
	    newEntries[ptr] = tmp;
	}
    }
    return std::unique_ptr<T[], Deleter>(newEntries, Deleter());
}

template <class T, class Deleter, class IntDeleter>
std::string SparseMatrixBase<T, Deleter, IntDeleter>::toString() const {
    std::stringstream stream;
    int entry = 0;
    int col = 0;
    for (int i = 0; i < dim(); i++) {
	int n = rowPtrs()[i+1] - rowPtrs()[i];
	stream << "[ ";
	for (int j = 0; j < dim(); j++) {
	    if (n <= 0 || j != cols()[col]) {
		stream << "0 ";
	    } else {
		stream << entries()[entry] << " ";
		col++;
		entry++;
               n--;
	    }
	}
	stream << "]\n";
    }
    return stream.str();
}

template <class T, class Deleter, class IntDeleter>
class MatVec {
private:
    const SparseMatrixBase<T, Deleter, IntDeleter>& mat;
    const DenseVector<T, Deleter>& vec;

public:
    MatVec(const SparseMatrixBase<T, Deleter, IntDeleter>& matrix, const DenseVector<T, Deleter>& vector): mat(matrix), vec(vector) { }
    
    int dim() const {
	return mat.dim();
    }

    void operator()(DenseVector<T, Deleter>& result) const {
	mat.matVec(vec, result);
    }
};

template <class T, class Deleter, class IntDeleter>
void SparseMatrixBase<T, Deleter, IntDeleter>::matVec(const DenseVector<T, Deleter>& vec, DenseVector<T, Deleter>& result) const {
    for (int i = 0; i < _dim; i++) {
	T tmp = 0;
	for (int j = rowPtrs()[i]; j < rowPtrs()[i+1]; j++) {
	    tmp += entries()[j] * vec[cols()[j]];
	}
	result[i] = tmp;
    }
}

template <class T, class Deleter, class IntDeleter>
MatVec<T, Deleter, IntDeleter> SparseMatrixBase<T, Deleter, IntDeleter>::operator*(const DenseVector<T, Deleter>& vec) const {
    return MatVec<T, Deleter, IntDeleter>(*this, vec);
}

template <class T, class Deleter>
template <class IntDeleter>
DenseVector<T, Deleter>::DenseVector(const MatVec<T, Deleter, IntDeleter>& op): _dim(op.dim()) {
    T* entriesPtr = Allocater<T, Deleter>::allocate(op.dim());
    _entries = std::unique_ptr<T[], Deleter>(entriesPtr, Deleter());
    op(*this);
}

template <class T, class Deleter>
template <class IntDeleter>
DenseVector<T, Deleter>& DenseVector<T, Deleter>::operator=(const MatVec<T, Deleter, IntDeleter>& op) {
    op(*this);
    return *this;
}

//Specialize for device matrices
#ifdef __CUDACC__
#include "gpu_memory.cuh"
#include "Parallel.cuh"

template <class T>
class SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>> {
private:
    int _dim;
    
public:
    SparseMatrixBase(int dim) : _dim(dim) { }
    
    virtual const std::unique_ptr<T[], gpu::CudaDeleter<T[]>>& entries() const = 0;

    virtual const std::unique_ptr<int[], gpu::CudaDeleter<int[]>>& cols() const = 0;

    virtual const std::unique_ptr<int[], gpu::CudaDeleter<int[]>>& rowPtrs() const = 0;
    
    int dim() const {
	return _dim;
    }

    int nonZeroEntries() const {
	int* dev = &rowPtrs()[_dim];
	int result = 0;
	gpu::memcpy_to_host<int>(&result, dev);
	return result;
    }

    std::unique_ptr<T[], gpu::CudaDeleter<T[]>> getAllDiagonals() const;

    std::unique_ptr<T[], gpu::CudaDeleter<T[]>> ssoraInverseEntries(double w) const;
    
    ProxySparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>> ssoraInverse(double w) const;
    
    std::string toString() const;

    void matVec(const DenseVector<T, gpu::CudaDeleter<T[]>>& vec, DenseVector<T, gpu::CudaDeleter<T[]>>& result) const;
    
    MatVec<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>> operator*(const DenseVector<T, gpu::CudaDeleter<T[]>>& vec) const;
};

template <class T>
std::unique_ptr<T[], gpu::CudaDeleter<T[]>> SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::getAllDiagonals() const {
    T* newEntries = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(dim());
    const int blocks = kernel::roundUpDiv(dim(),LinearAlgebra::threadsPerBlock);
    kernel::getAllDiagonals<<<blocks, LinearAlgebra::threadsPerBlock>>>(entries().get(),
									cols().get(),
									rowPtrs().get(),
									dim(),
									newEntries);
    return std::unique_ptr<T[], gpu::CudaDeleter<T[]>>(newEntries, gpu::CudaDeleter<T[]>());
}

template <class T>
std::unique_ptr<T[], gpu::CudaDeleter<T[]>>
SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::ssoraInverseEntries(double w) const {
    T* newEntries = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(nonZeroEntries());
    const int blocks = kernel::roundUpDiv(dim(),LinearAlgebra::threadsPerBlock);
    
    auto diags = getAllDiagonals();

    kernel::ssoraiEntries<<<blocks, LinearAlgebra::threadsPerBlock>>>(entries().get(),
								      cols().get(),
								      rowPtrs().get(),
								      diags.get(),
								      dim(),
								      w,
								      newEntries);
    return std::unique_ptr<T[], gpu::CudaDeleter<T[]>>(newEntries, gpu::CudaDeleter<T[]>());
}

template <class T>
void SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::matVec(const DenseVector<T, gpu::CudaDeleter<T[]>>& vec,
										 DenseVector<T, gpu::CudaDeleter<T[]>>& result) const {

    constexpr int warpSize = 32;
    const int blocks = kernel::roundUpDiv(dim(),LinearAlgebra::threadsPerBlock);
    const int sharedMemorySize = (LinearAlgebra::threadsPerBlock + warpSize/2)*sizeof(float) + warpSize * 2 * sizeof(int);
    kernel::sparseMatrixVectorProduct<T, warpSize><<<blocks, LinearAlgebra::threadsPerBlock, sharedMemorySize >>>(entries().get(),
														  cols().get(),
														  rowPtrs().get(),
														  vec.entries().get(),
														  dim(),
														  result.entries().get());
    checkCuda(cudaPeekAtLastError());
}

template <class T>
MatVec<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>
SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::operator*(const DenseVector<T, gpu::CudaDeleter<T[]>>& vec) const {
    return MatVec<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>(*this, vec);
}

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>::DenseVector(const MatVec<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& op): _dim(op.dim()) {
    T* entriesPtr = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(op.dim());
    _entries = std::unique_ptr<T[], gpu::CudaDeleter<T[]>>(entriesPtr, gpu::CudaDeleter<T[]>());
    op(*this);
}

template <class T>
DenseVector<T,  gpu::CudaDeleter<T[]>>&
DenseVector<T,  gpu::CudaDeleter<T[]>>::operator=(const MatVec<T,  gpu::CudaDeleter<T[]>,  gpu::CudaDeleter<int[]>>& op) {
    op(*this);
    return *this;
}

#endif

#endif
