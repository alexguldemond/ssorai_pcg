#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include "DenseVector.hpp"
#include <string>
#include <sstream>
#include <algorithm>
#include <utility>
#include <vector>
#include <iostream>

template<class T>
static T* allocate(int count) {
    return new T[count];
}

template <class T,
	  class Deleter = std::default_delete<T[]>,
	  class IntDeleter = std::default_delete<int[]>>
class SparseMatrix {
private:
    int _dim;
    std::function<T*(int)> _allocater;
    std::function<int*(int)> _intAllocater ;
public:
    
    std::unique_ptr<T[], Deleter> entries;
    std::unique_ptr<int[],IntDeleter> cols;
    std::unique_ptr<int[], IntDeleter> rowPtrs;
    
    SparseMatrix(int dim,
		 const std::unique_ptr<T[], Deleter>& newEntries,
		 const std::unique_ptr<int[], IntDeleter>& newCols,
		 const std::unique_ptr<int[], IntDeleter>& newRowPtrs,
		 const std::function<T*(int)>& allocater = allocate<T>,
		 const std::function<int*(int)>& intAllocater = allocate<int>): _dim(dim) {
	int nnz = newRowPtrs[dim];
	
	T* entriesPtr = allocater(nnz);
	std::copy(newEntries.get(), newEntries.get() + nnz, entriesPtr);
	
	int* colsPtr = intAllocater(nnz);
	std::copy(newCols.get(), newCols.get() + nnz, colsPtr);

	int* rowPtrsPtr = intAllocater(dim + 1);
	std::copy(newRowPtrs.get(), newRowPtrs.get() + dim + 1, rowPtrsPtr);

	entries = std::unique_ptr<T[], Deleter>(entriesPtr, Deleter());
	cols = std::unique_ptr<int[], IntDeleter>(colsPtr, IntDeleter());
	rowPtrs = std::unique_ptr<int[], IntDeleter>(rowPtrsPtr, IntDeleter());
	_allocater = allocater;
	_intAllocater = intAllocater;
    }

    SparseMatrix(int dim,
		 std::unique_ptr<T[], Deleter>&& newEntries,
		 std::unique_ptr<int[], IntDeleter>&& newCols,
		 std::unique_ptr<int[], IntDeleter>&& newRowPtrs,
		 const std::function<T*(int)&> allocater = allocate<T>,
		 const std::function<int*(int)>& intAllocater = allocate<int>): _dim(dim),
										_allocater(allocater),
										_intAllocater(intAllocater),
										entries(std::move(newEntries)),
										cols(std::move(newCols)),
										rowPtrs(std::move(newRowPtrs)) {

	
    }

    SparseMatrix(const SparseMatrix<T, Deleter, IntDeleter>& other) : _dim(other._dim) {
	int nnz = other.rowPtrs[_dim];
	    
	T* entriesPtr = other._allocater(nnz);
	std::copy(other.entries.get(), other.entries.get() + nnz, entriesPtr);
	
	int * colsPtr = other._intAllocater(nnz);
	std::copy(other.cols.get(), other.cols.get() + nnz, colsPtr);
	
	int* rowPtrsPtr = other._intAllocater(_dim + 1);
	std::copy(other.rowPtrs.get(), other.rowPtrs.get() + _dim + 1, rowPtrsPtr);

	entries = std::unique_ptr<T[], Deleter>(entriesPtr, Deleter());
	cols = std::unique_ptr<int[]>(colsPtr, IntDeleter());
	rowPtrs = std::unique_ptr<int[]>(rowPtrsPtr, IntDeleter());
	_allocater = other._allocater;
	_intAllocater = other._intAllocater;
    }

    SparseMatrix(SparseMatrix<T, Deleter, IntDeleter>&& other) : _dim(other._dim),
								 _allocater(std::move(other._allocater)),
								 _intAllocater(std::move(other._intAllocater)),
								 entries(std::move(other.entries)),
								 cols(std::move(other.cols)),
								 rowPtrs(std::move(other.rowPtrs)) {
	//Deliberately empty
    }
    
    SparseMatrix<T, Deleter, IntDeleter>& operator=(const SparseMatrix<T, Deleter, IntDeleter>& other) {
	if (this != other) {
	    _dim = other._dim;
	    int nnz = other.rowPtrs[_dim];
	    
	    T* entriesPtr = other._allocater(nnz);;
	    std::copy(other.entries.get(), other.entries.get() + nnz, entriesPtr);
	    
	    int * colsPtr = other._intAllocater(nnz);
	    std::copy(other.cols.get(), other.cols.get() + nnz, colsPtr);
	    
	    int* rowPtrsPtr =other._intAllocater(_dim + 1);
	    std::copy(other.rowPtrs.get(), other.rowPtrs.get() + _dim + 1, rowPtrsPtr);
	    
	    entries = std::unique_ptr<T[]>(entriesPtr, Deleter());
	    cols = std::unique_ptr<int[]>(colsPtr, IntDeleter());
	    rowPtrs = std::unique_ptr<int[]>(rowPtrsPtr, IntDeleter());
	    _allocater = other._allocater;
	    _intAllocater = other._intAllocater;
	}
	return *this;
    }

    SparseMatrix<T, Deleter, IntDeleter>& operator=(SparseMatrix<T, Deleter, IntDeleter>&& other) {
	if (this != &other) {
	    _dim = other._dim;
	    entries.swap(other.entries);
	    cols.swap(other.cols);
	    rowPtrs.swap(other.rowPtrs);
	}
	return *this;
    }
    
    int dim() const;

    int nonZeroEntries() const {
	return rowPtrs[_dim];
    }

    template<class VectorDeleter = std::default_delete<T[]> >
    DenseVector<T, VectorDeleter> operator*(const DenseVector<T, VectorDeleter>& vec) const;

    T operator()(int row, int col) const;
    
    T getDiagonal(int i) const;

    std::unique_ptr<T[], Deleter> getAllDiagonals() const;

    std::unique_ptr<T[], Deleter> ssoraInverseEntries(double w) const;
    
    SparseMatrix<T, Deleter, IntDeleter> ssoraInverse(double w) const;
    
    std::string toString() const;
    
};


template <class T, class Deleter, class IntDeleter>
inline int SparseMatrix<T, Deleter, IntDeleter>::dim() const {
    return _dim;
}

template <class T, class Deleter, class IntDeleter>
std::string SparseMatrix<T, Deleter, IntDeleter>::toString() const {
    std::stringstream stream;
    int entry = 0;
    int col = 0;
    for (int i = 0; i < dim(); i++) {
	int n = rowPtrs[i+1] - rowPtrs[i];
	stream << "[ ";
	for (int j = 0; j < dim(); j++) {
	    if (n <= 0 || j != cols[col]) {
		stream << "0 ";
	    } else {
		stream << entries[entry] << " ";
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
template <class VectorDeleter>
DenseVector<T, VectorDeleter> SparseMatrix<T, Deleter, IntDeleter>::operator*(const DenseVector<T, VectorDeleter>& vec) const {
    T* newEntries = _allocater(dim());
    std::fill(newEntries, newEntries + dim(), 0);
    for (int i = 0; i < _dim; i++) {
	for (int j = rowPtrs[i]; j < rowPtrs[i+1]; j++) {
	    newEntries[i] += entries[j] * vec(cols[j]);
	}
    }
    DenseVector<T, VectorDeleter> newVec(_dim, std::unique_ptr<T[], VectorDeleter>(newEntries, VectorDeleter()));
    return newVec;
}

template <class T, class Deleter, class IntDeleter>
T SparseMatrix<T, Deleter, IntDeleter>::operator()(int row, int col) const {
    for (int j = rowPtrs[row]; j < rowPtrs[row+1]; j++) {
	if (cols[j] == col) {
	    return entries[j];
	}
    }
    return 0;
}

template <class T, class Deleter, class IntDeleter>
T SparseMatrix<T, Deleter, IntDeleter>::getDiagonal(int i) const {
    int start = rowPtrs[i];
    int end = rowPtrs[i+1];
    int middle = (start + end)/2;
    while (cols[middle] != i) {
	if (i < cols[middle]) {
	    end = middle;
	} else {
	    start = middle;
	}
	middle = (start + end)/2;
    }
    return entries[middle];
}

template <class T, class Deleter, class IntDeleter>
std::unique_ptr<T[], Deleter> SparseMatrix<T, Deleter, IntDeleter>::getAllDiagonals() const {
    T* diagonals = _allocater(dim());
    for (int i = 0; i < dim(); i++) {
	diagonals[i] = getDiagonal(i);
    }
    return std::unique_ptr<T[], Deleter>(diagonals, Deleter());
}


template <class T, class Deleter, class IntDeleter>
std::unique_ptr<T[], Deleter> SparseMatrix<T, Deleter, IntDeleter>::ssoraInverseEntries(double w) const {
    double mult = w*(2 - w);
    int nnz = nonZeroEntries();
    T* newEntries = _allocater(nnz);
    auto diagonals = getAllDiagonals();
    for (int i = 0; i < dim(); i++) {
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
	    while (iPtr < rowPtrs[row + 1] && jPtr < rowPtrs[col + 1]) {
		while(cols[jPtr] <= col && jPtr < nnz) {
		    jPtr++;
		}

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
	    
	    tmp *= w*w / (diagonals[row] * diagonals[col]);
	    tmp += (i == j) ? 1.0/diagonals[row] : -1.0 * w * entries[ptr] / (diagonals[row] * diagonals[col]);
	    tmp *= mult;
	    newEntries[ptr] = tmp;
	}
    }
    return std::unique_ptr<T[], Deleter>(newEntries, Deleter());
}


template <class T, class Deleter, class IntDeleter>
SparseMatrix<T, Deleter, IntDeleter> SparseMatrix<T, Deleter, IntDeleter>::ssoraInverse(double w) const {
    int* newCols = _intAllocater(nonZeroEntries());
    std::copy(cols.get(), cols.get() + nonZeroEntries(), newCols);
    int* newRowPtrs = _intAllocater(_dim + 1);
    std::copy(rowPtrs.get(), rowPtrs.get() + _dim + 1, newRowPtrs);
    SparseMatrix m(_dim,
		   ssoraInverseEntries(w),
		   std::unique_ptr<int[], IntDeleter>(newCols, IntDeleter()),
		   std::unique_ptr<int[], IntDeleter>(newRowPtrs, IntDeleter()),
		   _allocater,
		   _intAllocater);
    return m;
}
namespace SparseMatrixFactory {

    template <class T, class Deleter, class IntDeleter >
    SparseMatrix<T, Deleter, IntDeleter> triDiagonal(int dim, T left, T middle, T right,
						     const Deleter& deleter,
						     const IntDeleter& intDeleter,
						     const std::function<T*(int)>& allocater = allocate<T>,
						     const std::function<int*(int)>& intAllocater = allocate<int>) {
	T* entries;
	int* cols;
	int* rowPtrs;
	
	if (dim == 1) {
	    entries = allocater(1);
	    entries[0] = middle;
	    cols = intAllocater(1);
	    cols[0] = 0;
	    rowPtrs = intAllocater(2);
	    rowPtrs[0] = 0;
	    rowPtrs[1] = 1;
	} else if (dim == 2) {
	    entries = allocater(4);
	    entries[0] = middle;
	    entries[1] = right;
	    entries[2] = left;
	    entries[3] = middle;
	    cols = intAllocater(4);
	    cols[0] = cols[2] = 0;
	    cols[1] = cols[3] = 1;
	    rowPtrs = intAllocater(3);
	    rowPtrs[0] = 0;
	    rowPtrs[1] = 2;
	    rowPtrs[2] = 4;
	} else {
	    int nonZero = 4 + 3 * (dim - 2);
	    entries = allocater(nonZero);
	    cols = intAllocater(nonZero);
	    rowPtrs = intAllocater(dim + 1);
	    
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
	SparseMatrix<T, Deleter, IntDeleter> m(dim,
					       std::unique_ptr<T[], Deleter>(entries, deleter),
					       std::unique_ptr<int[], IntDeleter>(cols, intDeleter),
					       std::unique_ptr<int[], IntDeleter>(rowPtrs, intDeleter),
					       allocater,
					       intAllocater);
	return m;
    }

    template <class T>
    SparseMatrix<T> triDiagonal(int dim, T left, T middle, T right) {
	return triDiagonal<T,std::default_delete<T[]>, std::default_delete<int[]>>(dim, left, middle, right, std::default_delete<T[]>(), std::default_delete<int[]>());
    }

    template <class T, class Deleter, class IntDeleter >
    SparseMatrix<T, Deleter, IntDeleter> bandMatrix(std::size_t dim,
						    const std::vector<T>& band,
						    const Deleter& deleter,
						    const IntDeleter& intDeleter,
						    const std::function<T*(int)>& allocater = allocate<T>,
						    const std::function<int*(int)>& intAllocater = allocate<int>) {
	
	std::size_t maxEntries = 2*band.size() - 1;
	std::size_t nonFullRows = maxEntries - 1;
	std::size_t fullRows = dim - nonFullRows;
	std::size_t nnz = fullRows*maxEntries +2 * ( maxEntries * (maxEntries - 1)/2 - ( band.size()*(band.size() - 1) )/2);
	
	T* entries = allocater(nnz);
	int* cols = intAllocater(nnz);
	int* rowPtrs = intAllocater(dim + 1);
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
	
	SparseMatrix<T, Deleter, IntDeleter> m(dim,
					       std::unique_ptr<T[], Deleter>(entries, deleter),
					       std::unique_ptr<int[], IntDeleter>(cols, intDeleter),
					       std::unique_ptr<int[], IntDeleter>(rowPtrs, intDeleter),
					       allocater,
					       intAllocater);
	return m;
    }

    template <class T>
    SparseMatrix<T> bandMatrix(int dim, const std::vector<T>& band) {
	return  bandMatrix<T, std::default_delete<T[]>, std::default_delete<int[]>>(dim, band, std::default_delete<T[]>(), std::default_delete<int[]>());
    }
    
}


#endif
