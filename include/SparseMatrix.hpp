#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include "DenseVector.hpp"
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>


template <class T>
class SparseMatrix {
public:
    std::vector<T> entries;
    std::vector<int> cols;
    std::vector<int> rowPtrs;
    
    SparseMatrix(const std::vector<T>& entries,
		 const std::vector<int>& columnIndices,
		 const std::vector<int>& compressedRow): entries(entries),
							 cols(columnIndices),
							 rowPtrs(compressedRow) {
	//Deliberately empty
    }

    SparseMatrix(std::vector<T>&& entries,
		 std::vector<int>&& columnIndices,
		 std::vector<int>&& compressedRow): entries(entries),
						    cols(columnIndices),
						    rowPtrs(compressedRow) {
	//Deliberately empty
    }
    
    static SparseMatrix triDiagonal(int dim, T left, T middle, T right);
    
    int dim() const;

    int nonZeroEntries() const {
	return rowPtrs[dim()];
    }

    DenseVector<T> operator*(const DenseVector<T>& vec) const;

    T operator()(int row, int col) const;
    
    T getDiagonal(int i) const;

    std::vector<T> getAllDiagonals() const;
    
    SparseMatrix<T> ssoraInverse(double w) const;
    
    std::string toString() const;
    
};


template <class T>
int SparseMatrix<T>::dim() const {
    return rowPtrs.size() - 1;
}

template <class T>
std::string SparseMatrix<T>::toString() const {
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

template <class T>
DenseVector<T> SparseMatrix<T>::operator*(const DenseVector<T>& vec) const {
    int dim = vec.dim();
    std::vector<T> newEntries(dim, 0);
    DenseVector<T> newVec(newEntries);

    for (int i = 0; i < dim; i++) {
	for (int j = rowPtrs[i]; j < rowPtrs[i+1]; j++) {
	    newVec[i] += entries[j] * vec(cols[j]);
	}
    }
    return newVec;
}

template <class T>
T SparseMatrix<T>::operator()(int row, int col) const {
    for (int j = rowPtrs[row]; j < rowPtrs[row+1]; j++) {
	if (cols[j] == col) {
	    return entries[j];
	}
    }
    return 0;
}

template <class T>
T SparseMatrix<T>::getDiagonal(int i) const {
    T d;
    for (int j = rowPtrs[i]; j < rowPtrs[i+1]; j++) {
	if (cols[j] == i) {
	    d = entries[j];
	    break;
	}
    }
    return d;
}

//TODO lazily
template <class T>
std::vector<T> SparseMatrix<T>::getAllDiagonals() const {
    std::vector<T> diagonals(dim());
    for (int i = 0; i < dim(); i++) {
	diagonals[i] = getDiagonal(i);
    }
    return diagonals;
}


template <class T>
SparseMatrix<T> SparseMatrix<T>::ssoraInverse(double w) const {
    double mult = w*(2 - w);
    std::vector<T> newEntries(nonZeroEntries());
    std::vector<int> newCols(cols);
    std::vector<int> newRowPtrs(rowPtrs);
    
    std::vector<T> diagonals = getAllDiagonals();
    
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
	      
	    for (int iPtr = rowPtrs[row]; iPtr < rowPtrs[row+1]; iPtr++) {
		for (int jPtr = rowPtrs[col]; jPtr < rowPtrs[col+1]; jPtr++) {
		    if (cols[jPtr] == cols[iPtr] && cols[jPtr] > col) {
			newEntries[ptr] += entries[jPtr] * entries[iPtr] / diagonals[cols[jPtr]];
		    }
		}
	    }
	    newEntries[ptr] *= w*w / (diagonals[row] * diagonals[col]);
	    newEntries[ptr] += (i == j) ? 1.0/diagonals[row] : -1.0 * w * entries[ptr] / (diagonals[row] * diagonals[col]);
	    newEntries[ptr] *= mult;   
	}
    }
    SparseMatrix m(newEntries, newCols, newRowPtrs);
    
    return m;
}

template <class T>
SparseMatrix<T> SparseMatrix<T>::triDiagonal(int dim, T left, T middle, T right) {
    std::vector<T>  entries;
    std::vector<int> cols;
    std::vector<int> rowPtrs;

    if (dim == 1) {
	entries = std::vector<T> {middle};
	cols = std::vector<int>{ 0 };
	rowPtrs = std::vector<int>{0, 1};
    } else if (dim == 2) {
	entries = std::vector<T>{middle, right, left, middle};
	cols = std::vector<int>{0, 1, 0, 1};
	rowPtrs = std::vector<int>{0, 2, 4};
    } else {
	int nonZero = 4 + 3 * (dim - 2);
	entries = std::vector<T>(nonZero);
	cols = std::vector<int>(nonZero);;
	rowPtrs = std::vector<int>(dim + 1);

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
    SparseMatrix<T> m(entries, cols, rowPtrs);
    return m;
}


#endif
