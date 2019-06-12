#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include "DenseVector.hpp"
#include "SparseMatrixBase.hpp"
#include <string>
#include <sstream>
#include <algorithm>
#include <utility>
#include <vector>

template <class T,
	  class Deleter = std::default_delete<T[]>,
	  class IntDeleter = std::default_delete<int[]>>
class SparseMatrix: public SparseMatrixBase<T, Deleter, IntDeleter> {
private:
    std::unique_ptr<T[], Deleter> _entries;
    std::unique_ptr<int[],IntDeleter> _cols;
    std::unique_ptr<int[], IntDeleter> _rowPtrs;
public: 
    
    SparseMatrix(int dim,
		 const std::unique_ptr<T[], Deleter>& newEntries,
		 const std::unique_ptr<int[], IntDeleter>& newCols,
		 const std::unique_ptr<int[], IntDeleter>& newRowPtrs);

    SparseMatrix(int dim,
		 std::unique_ptr<T[], Deleter>&& newEntries,
		 std::unique_ptr<int[], IntDeleter>&& newCols,
		 std::unique_ptr<int[], IntDeleter>&& newRowPtrs);

    SparseMatrix(SparseMatrix<T, Deleter, IntDeleter>&& other);
    
    SparseMatrix(const SparseMatrix<T, Deleter, IntDeleter>& other);
    
    SparseMatrix<T, Deleter, IntDeleter>& operator=(const SparseMatrix<T, Deleter, IntDeleter>& other);

    SparseMatrix<T, Deleter, IntDeleter>& operator=(SparseMatrix<T, Deleter, IntDeleter>&& other);

    const std::unique_ptr<T[], Deleter>& entries() const override {
	return _entries;
    }

    const std::unique_ptr<int[], IntDeleter>& cols() const override {
	return _cols;
    }

    const std::unique_ptr<int[], IntDeleter>& rowPtrs() const override {
	return _rowPtrs;
    }

    static SparseMatrix<T, Deleter, IntDeleter> triDiagonal(int dim, T left, T middle, T right);

    static SparseMatrix<T, Deleter, IntDeleter> bandMatrix(std::size_t dim, const std::vector<T>& band);
};

template <class T, class Deleter, class IntDeleter>
SparseMatrix<T, Deleter, IntDeleter>::SparseMatrix(int dim,
						   const std::unique_ptr<T[], Deleter>& newEntries,
						   const std::unique_ptr<int[], IntDeleter>& newCols,
						   const std::unique_ptr<int[], IntDeleter>& newRowPtrs) : SparseMatrixBase<T, Deleter, IntDeleter>(dim) {
    int nnz = newRowPtrs[dim];
    
    T* entriesPtr = Allocater<T, Deleter>::allocate(nnz);
    std::copy(newEntries.get(), newEntries.get() + nnz, entriesPtr);
    
    int* colsPtr = Allocater<int, IntDeleter>::allocate(nnz);
    std::copy(newCols.get(), newCols.get() + nnz, colsPtr);
    
    int* rowPtrsPtr = Allocater<int, IntDeleter>::allocate(dim + 1);
    std::copy(newRowPtrs.get(), newRowPtrs.get() + dim + 1, rowPtrsPtr);
    
    _entries = std::unique_ptr<T[], Deleter>(entriesPtr, Deleter());
    _cols = std::unique_ptr<int[], IntDeleter>(colsPtr, IntDeleter());
    _rowPtrs = std::unique_ptr<int[], IntDeleter>(rowPtrsPtr, IntDeleter());
}

template <class T, class Deleter, class IntDeleter>
SparseMatrix<T, Deleter, IntDeleter>::SparseMatrix(int dim,
						   std::unique_ptr<T[], Deleter>&& newEntries,
						   std::unique_ptr<int[], IntDeleter>&& newCols,
						   std::unique_ptr<int[], IntDeleter>&& newRowPtrs): SparseMatrixBase<T, Deleter, IntDeleter>(dim),
												  _entries(std::move(newEntries)),
												  _cols(std::move(newCols)),
												  _rowPtrs(std::move(newRowPtrs)) {
    
    
}


template <class T, class Deleter, class IntDeleter>
SparseMatrix<T, Deleter, IntDeleter>::SparseMatrix(const SparseMatrix<T, Deleter, IntDeleter>& other) : SparseMatrixBase<T, Deleter, IntDeleter>(other) {
    int nnz = other.rowPtrs[other.dim()];
    
    T* entriesPtr = Allocater<T, Deleter>::allocate(nnz);
    std::copy(other.entries.get(), other.entries.get() + nnz, entriesPtr);
    
    int * colsPtr = Allocater<int, IntDeleter>::allocate(nnz);
    std::copy(other.cols.get(), other.cols.get() + nnz, colsPtr);
    
    int* rowPtrsPtr = Allocater<int, IntDeleter>::allocate(other.dim() + 1);
    std::copy(other.rowPtrs.get(), other.rowPtrs.get() + other.dim() + 1, rowPtrsPtr);
    
    _entries = std::unique_ptr<T[], Deleter>(entriesPtr, Deleter());
    _cols = std::unique_ptr<int[], IntDeleter>(colsPtr, IntDeleter());
    _rowPtrs = std::unique_ptr<int[], IntDeleter>(rowPtrsPtr, IntDeleter());
}

template <class T, class Deleter, class IntDeleter>
SparseMatrix<T, Deleter, IntDeleter>::SparseMatrix(SparseMatrix<T, Deleter, IntDeleter>&& other) : SparseMatrixBase<T, Deleter, IntDeleter>(other),
												   _entries(std::move(other._entries)),
												   _cols(std::move(other._cols)),
												   _rowPtrs(std::move(other._rowPtrs)) {
    //Deliberately empty
}

template <class T, class Deleter, class IntDeleter>
SparseMatrix<T, Deleter, IntDeleter>& SparseMatrix<T, Deleter, IntDeleter>::operator=(const SparseMatrix<T, Deleter, IntDeleter>& other) {
    if (this != other) {
	SparseMatrixBase<T, Deleter, IntDeleter>::operator=(other);
	int nnz = other.rowPtrs[other.dim()];
	
	T* entriesPtr = Allocater<T, Deleter>::allocate(nnz);
	std::copy(other.entries.get(), other.entries.get() + nnz, entriesPtr);
	
	int * colsPtr = Allocater<T, Deleter>::allocate(nnz);
	std::copy(other.cols.get(), other.cols.get() + nnz, colsPtr);
	
	int* rowPtrsPtr = Allocater<T, Deleter>::allocate(other.dim() + 1);
	std::copy(other.rowPtrs.get(), other.rowPtrs.get() + other.dim() + 1, rowPtrsPtr);
	
	_entries = std::unique_ptr<T[], Deleter>(entriesPtr, Deleter());
	_cols = std::unique_ptr<int[], IntDeleter>(colsPtr, IntDeleter());
	_rowPtrs = std::unique_ptr<int[], IntDeleter>(rowPtrsPtr, IntDeleter());
    }
    return *this;
}


template <class T, class Deleter, class IntDeleter>
SparseMatrix<T, Deleter, IntDeleter>& SparseMatrix<T, Deleter, IntDeleter>::operator=(SparseMatrix<T, Deleter, IntDeleter>&& other) {
    if (this != &other) {
	SparseMatrixBase<T, Deleter, IntDeleter>::operator=(other);
	_entries.swap(other._entries);
	_cols.swap(other._cols);
	_rowPtrs.swap(other._rowPtrs);
    }
    return *this;
}

template <class T, class Deleter = std::default_delete<T[]>, class IntDeleter = std::default_delete<int[]> >
class ProxySparseMatrix: public SparseMatrixBase<T, Deleter, IntDeleter> {
private:
    std::unique_ptr<T[], Deleter> _entries;
    const SparseMatrixBase<T, Deleter, IntDeleter>& _mat;
    
public:
    ProxySparseMatrix(const std::unique_ptr<T[], Deleter>& newEntries,
		      const SparseMatrixBase<T, Deleter, IntDeleter>& mat): SparseMatrixBase<T, Deleter, IntDeleter>(mat.dim()),
										    _entries(newEntries),
										    _mat(mat) { }

    ProxySparseMatrix(std::unique_ptr<T[], Deleter>&& newEntries,
		      const SparseMatrixBase<T, Deleter, IntDeleter>& mat): SparseMatrixBase<T, Deleter, IntDeleter>(mat.dim()),
										    _entries(std::move(newEntries)),
										    _mat(mat) { }
    
    const std::unique_ptr<T[], Deleter>& entries() const override {
	return _entries;
    }
    
    const std::unique_ptr<int[], IntDeleter>& cols() const override {
	return _mat.cols();
    }

    const std::unique_ptr<int[], IntDeleter>& rowPtrs() const override {
	return _mat.rowPtrs();
    }
};

template <class T, class Deleter, class IntDeleter>
ProxySparseMatrix<T, Deleter, IntDeleter>  SparseMatrixBase<T, Deleter, IntDeleter>::ssoraInverse(T w) const {
    return ProxySparseMatrix<T, Deleter, IntDeleter>(ssoraInverseEntries(w), (*this));
}


template <class T, class Deleter, class IntDeleter >
SparseMatrix<T, Deleter, IntDeleter> SparseMatrix<T, Deleter, IntDeleter>::triDiagonal(int dim, T left, T middle, T right) {
    T* entries;
    int* cols;
    int* rowPtrs;
    
    if (dim == 1) {
	entries = Allocater<T, Deleter>::allocate(1);
	entries[0] = middle;
	cols = Allocater<int, IntDeleter>::allocate(1);
	cols[0] = 0;
	rowPtrs = Allocater<int, IntDeleter>::allocate(2);
	rowPtrs[0] = 0;
	rowPtrs[1] = 1;
    } else if (dim == 2) {
	entries = Allocater<T,Deleter>::allocate(4);
	entries[0] = middle;
	entries[1] = right;
	entries[2] = left;
	entries[3] = middle;
	cols = Allocater<int, IntDeleter>::allocate(4);
	cols[0] = cols[2] = 0;
	cols[1] = cols[3] = 1;
	rowPtrs = Allocater<int, IntDeleter>::allocate(3);
	rowPtrs[0] = 0;
	rowPtrs[1] = 2;
	rowPtrs[2] = 4;
    } else {
	int nonZero = 3 * dim - 2;
	entries = Allocater<T, Deleter>::allocate(nonZero);
	cols = Allocater<int, IntDeleter>::allocate(nonZero);
	rowPtrs = Allocater<int, IntDeleter>::allocate(dim + 1);
	
	entries[0] = middle;
	entries[1] = right;
	cols[0] = 0;
	cols[1] = 1;
	rowPtrs[0] = 0;
	rowPtrs[1] = 2;
	    
	for (int i = 0; i < dim - 2; i++) {
	    entries[2 + 3*i] = left;
	    entries[3 + 3*i] = middle;
	    entries[4 + 3*i] = right;
	    cols[2 + 3*i] = i;
	    cols[3 + 3*i] = i + 1;
	    cols[4 + 3*i] = i + 2;
	    rowPtrs[i+2] = 2 + 3*(i + 1);
	}
	entries[nonZero - 2] = left;
	entries[nonZero - 1] = middle;
	cols[nonZero - 2] = dim - 2;
	cols[nonZero - 1] = dim -1;
	rowPtrs[dim] = rowPtrs[dim - 1] + 2;
    }
    return SparseMatrix<T, Deleter, IntDeleter>(dim,
						std::unique_ptr<T[], Deleter>(entries, Deleter()),
						std::unique_ptr<int[], IntDeleter>(cols, IntDeleter()),
						std::unique_ptr<int[], IntDeleter>(rowPtrs, IntDeleter()));
    
}


template <class T, class Deleter, class IntDeleter >
SparseMatrix<T, Deleter, IntDeleter> SparseMatrix<T, Deleter, IntDeleter>::bandMatrix(std::size_t dim,
										      const std::vector<T>& band) {
    std::size_t maxEntries = 2*band.size() - 1;
    std::size_t nonFullRows = maxEntries - 1;
    std::size_t fullRows = dim - nonFullRows;
    std::size_t nnz = fullRows*maxEntries +2 * ( maxEntries * (maxEntries - 1)/2 - ( band.size()*(band.size() - 1) )/2);
    
    T* entries = Allocater<T, Deleter>::allocate(nnz);
    int* cols = Allocater<int, IntDeleter>::allocate(nnz);
    int* rowPtrs = Allocater<int, IntDeleter>::allocate(dim + 1);
    rowPtrs[0] = 0;
    
    //First set of non full rows
    std::size_t j = 0;
    std::size_t ptr = 1;
    for (std::size_t row = 0; row < nonFullRows/2; row++) {
	//Left of diagonal
	
	for (std::size_t col = 0; col < row; col++) {
	    entries[j] = band[row - col];
	    cols[j] = col;
	    j++;
	}
	//Diagonal and right of diagonal
	for (std::size_t col = row; col < band.size() + row; col++) {
	    entries[j] = band[col - row ];	
	    cols[j] = col;
	    j++;
	}
	rowPtrs[ptr] = row + band.size() + rowPtrs[ptr - 1];
	ptr++;
    }
    
    //Full rows
    for (std::size_t row = nonFullRows/2; row < nonFullRows/2 + fullRows; row++) {
	for (std::size_t col = row - band.size() + 1; col < row; col++) {
	    entries[j] = band[row - col];
	    cols[j] = col;
	    j++;
	}
	for (std::size_t col = row; col < band.size() + row; col++) {
	    entries[j] = band[col - row];
	    cols[j] = col;
	    j++;
	}
	rowPtrs[ptr] = maxEntries + rowPtrs[ptr - 1];
	ptr++;
    }
    
    //Ending batch of non full rows
    for (std::size_t row = nonFullRows/2 + fullRows; row < dim; row++) {
	for (std::size_t col = row - band.size() + 1; col < row; col++) {
	    entries[j] = band[row - col];
	    cols[j] = col;
	    j++;
	}
	for (std::size_t col = row; col < dim; col++) {
	    entries[j] = band[col - row];
	    cols[j] = col;
	    j++;
	}
	rowPtrs[ptr] = (dim - row - 1) + band.size() + rowPtrs[ptr - 1];
	ptr++;
    }
    
    return SparseMatrix<T, Deleter, IntDeleter>(dim,
						std::unique_ptr<T[], Deleter>(entries, Deleter()),
						std::unique_ptr<int[], IntDeleter>(cols, IntDeleter()),
						std::unique_ptr<int[], IntDeleter>(rowPtrs, IntDeleter()));
}


//Device specializations
#ifdef __CUDACC__

template <class T>
std::string SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::toString() const {
    int nnz = nonZeroEntries();
    std::unique_ptr<T[]> newEntries = gpu::get_from_device<T>(entries(), nnz);
    std::unique_ptr<int[]> newCols = gpu::get_from_device<int>(cols(), nnz);
    std::unique_ptr<int[]> newRowPtrs = gpu::get_from_device<int>(rowPtrs(), dim() + 1);
    return SparseMatrix<T>(dim(), newEntries, newCols, newRowPtrs).toString();
}

template <class T>
class SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>: public SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>> {
private:
    std::unique_ptr<T[], gpu::CudaDeleter<T[]>> _entries;
    std::unique_ptr<int[], gpu::CudaDeleter<int[]>> _cols;
    std::unique_ptr<int[], gpu::CudaDeleter<int[]>> _rowPtrs;
public: 
    
    SparseMatrix(int dim,
		 const std::unique_ptr<T[], gpu::CudaDeleter<T[]>>& newEntries,
		 const std::unique_ptr<int[], gpu::CudaDeleter<T[]>>& newCols,
		 const std::unique_ptr<int[], gpu::CudaDeleter<T[]>>& newRowPtrs);

    SparseMatrix(int dim,
		 std::unique_ptr<T[], gpu::CudaDeleter<T[]>>&& newEntries,
		 std::unique_ptr<int[], gpu::CudaDeleter<int[]>>&& newCols,
		 std::unique_ptr<int[], gpu::CudaDeleter<int[]>>&& newRowPtrs);

    SparseMatrix(SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>&& other);
    
    SparseMatrix(const SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& other);
    
    SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& operator=(const SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& other);

    SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& operator=(SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>&& other);

    const std::unique_ptr<T[], gpu::CudaDeleter<T[]>>& entries() const override {
	return _entries;
    }

    const std::unique_ptr<int[], gpu::CudaDeleter<int[]>>& cols() const override {
	return _cols;
    }

    const std::unique_ptr<int[], gpu::CudaDeleter<int[]>>& rowPtrs() const override {
	return _rowPtrs;
    }

    static SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>> triDiagonal(int dim, T left, T middle, T right);

    static SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>> bandMatrix(std::size_t dim, const std::vector<T>& band);
};


template <class T>
ProxySparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>  SparseMatrixBase<T,gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::ssoraInverse(T w) const {
    return ProxySparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>(ssoraInverseEntries(w), (*this));
}

template <class T>
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::SparseMatrix(int dim,
									      const std::unique_ptr<T[], gpu::CudaDeleter<T[]>>& newEntries,
									      const std::unique_ptr<int[], gpu::CudaDeleter<T[]>>& newCols,
									      const std::unique_ptr<int[], gpu::CudaDeleter<T[]>>& newRowPtrs):
    SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>(dim) {
    
    int* dev = &newRowPtrs()[dim];
    int nnz = 0;
    gpu::memcpy_to_host<int>(&nnz, dev);

    int blocks = LinearAlgebra::blocks;
    
    T* entriesPtr = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(nnz);
    kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(newEntries.get(), entriesPtr, nnz);
    
    int* colsPtr = Allocater<int, gpu::CudaDeleter<int[]>>::allocate(nnz);
    kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(newCols.get(), colsPtr, nnz);
    
    int* rowPtrsPtr = Allocater<int, gpu::CudaDeleter<int[]>>::allocate(dim + 1);
    blocks = kernel::roundUpDiv(dim + 1, LinearAlgebra::threadsPerBlock);
    kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(newRowPtrs.get(), rowPtrsPtr, nnz);
    
    _entries = std::unique_ptr<T[], gpu::CudaDeleter<T[]>>(entriesPtr, gpu::CudaDeleter<T[]>());
    _cols = std::unique_ptr<int[], gpu::CudaDeleter<int[]>>(colsPtr, gpu::CudaDeleter<int[]>());
    _rowPtrs = std::unique_ptr<int[], gpu::CudaDeleter<int[]>>(rowPtrsPtr, gpu::CudaDeleter<int[]>());
}

template <class T>
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::SparseMatrix(int dim,
								std::unique_ptr<T[], gpu::CudaDeleter<T[]>>&& newEntries,
								std::unique_ptr<int[], gpu::CudaDeleter<int[]>>&& newCols,
								std::unique_ptr<int[], gpu::CudaDeleter<int[]>>&& newRowPtrs):
    SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>(dim),
    _entries(std::move(newEntries)),
    _cols(std::move(newCols)),
    _rowPtrs(std::move(newRowPtrs)) {
}


template <class T>
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::SparseMatrix(const SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& other):
    SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>(other) {

    int nnz = other.nonZeroEntries();
    
    int blocks = LinearAlgebra::blocks;
    
    T* entriesPtr = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(nnz);
    kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(other.entries().get(), entriesPtr, nnz);
    
    int* colsPtr = Allocater<int, gpu::CudaDeleter<int[]>>::allocate(nnz);
    kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(other.cols().get(), colsPtr, nnz);
    
    int* rowPtrsPtr = Allocater<int, gpu::CudaDeleter<int[]>>::allocate(other.dim() + 1);
    blocks = kernel::roundUpDiv(other.dim() + 1, LinearAlgebra::threadsPerBlock);
    kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(other.rowPtr().get(), rowPtrsPtr, nnz);
    
    _entries = std::unique_ptr<T[], gpu::CudaDeleter<T[]>>(entriesPtr, gpu::CudaDeleter<T[]>());
    _cols = std::unique_ptr<int[], gpu::CudaDeleter<int[]>>(colsPtr, gpu::CudaDeleter<int[]>());
    _rowPtrs = std::unique_ptr<int[], gpu::CudaDeleter<int[]>>(rowPtrsPtr, gpu::CudaDeleter<int[]>());
}
    
template <class T>
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::SparseMatrix(SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>&& other) :
    SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>(other),
    _entries(std::move(other._entries)),
    _cols(std::move(other._cols)),
    _rowPtrs(std::move(other._rowPtrs)) {
    //Deliberately empty
}

template <class T>
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>&
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::operator=(const SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& other) {
    if (this != other) {
	SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::operator=(other);
	int nnz = other.nonZeroEntries();
    
	int blocks = LinearAlgebra::blocks;
    
	T* entriesPtr = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(nnz);
	kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(other.entries().get(), entriesPtr, nnz);
	
	int* colsPtr = Allocater<int, gpu::CudaDeleter<int[]>>::allocate(nnz);
	kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(other.cols().get(), colsPtr, nnz);
	
	int* rowPtrsPtr = Allocater<int, gpu::CudaDeleter<int[]>>::allocate(other.dim() + 1);
	blocks = kernel::roundUpDiv(other.dim() + 1, LinearAlgebra::threadsPerBlock);
	kernel::copyArray<<<blocks, LinearAlgebra::threadsPerBlock>>>(other.rowPtrs().get(), rowPtrsPtr, nnz);
	
	_entries = std::unique_ptr<T[], gpu::CudaDeleter<T[]>>(entriesPtr, gpu::CudaDeleter<T[]>());
	_cols = std::unique_ptr<int[], gpu::CudaDeleter<int[]>>(colsPtr, gpu::CudaDeleter<int[]>());
	_rowPtrs = std::unique_ptr<int[], gpu::CudaDeleter<int[]>>(rowPtrsPtr, gpu::CudaDeleter<int[]>());
    }
    return *this;
}


template <class T>
SparseMatrix<T,gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>&
SparseMatrix<T,gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::operator=(SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>&& other) {
    if (this != &other) {
	SparseMatrixBase<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::operator=(other);
	_entries.swap(other._entries);
	_cols.swap(other._cols);
	_rowPtrs.swap(other._rowPtrs);
    }
    return *this;
}

template <class T>
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::triDiagonal(int dim, T left, T middle, T right) {
    const int nnz = 3*dim - 2;

    T* entries = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(nnz);
    int* cols = Allocater<int, gpu::CudaDeleter<int[]>>::allocate(nnz);
    int* rowPtrs = Allocater<int, gpu::CudaDeleter<int[]>>::allocate(dim + 1);
    
    int blocks = LinearAlgebra::blocks;
    kernel::triDiagonal<T><<< blocks, LinearAlgebra::threadsPerBlock >>>(dim , left, middle, right, entries, cols, rowPtrs);
    checkCuda(cudaPeekAtLastError());
    return SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>(dim,
									   std::unique_ptr<T[], gpu::CudaDeleter<T[]>>(entries, gpu::CudaDeleter<T[]>()),
									   std::unique_ptr<int[], gpu::CudaDeleter<int[]>>(cols, gpu::CudaDeleter<int[]>()),
									   std::unique_ptr<int[], gpu::CudaDeleter<int[]>>(rowPtrs, gpu::CudaDeleter<int[]>()));
    
}

template <class T >
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>
SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>::bandMatrix(std::size_t dim,
									    const std::vector<T>& band) {
    auto mat = SparseMatrix<T, gpu::CudaHostDeleter<T[]>, gpu::CudaHostDeleter<int[]>>::bandMatrix(dim, band);
    
    int nnz = mat.nonZeroEntries();

    std::unique_ptr<T[], gpu::CudaDeleter<T[]>> entries = gpu::make_device<float>(nnz);
    std::unique_ptr<int[], gpu::CudaDeleter<int[]>> cols = gpu::make_device<int>(nnz);
    std::unique_ptr<int[], gpu::CudaDeleter<int[]>> rowPtrs = gpu::make_device<int>(dim + 1);
    
    gpu::memcpy_to_device<float>(mat.entries(), entries, nnz);
    gpu::memcpy_to_device<int>(mat.cols(), cols, nnz);
    gpu::memcpy_to_device<int>(mat.rowPtrs(), rowPtrs, dim + 1);
    
    return SparseMatrix<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>(dim, std::move(entries), std::move(cols), std::move(rowPtrs));
}






#endif


#endif
