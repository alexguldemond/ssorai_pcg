#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include <iostream>
#include <vector>

typedef double Number;

int main() {
    
    std::vector<Number> band(100, -1);
    band[0] = 40;
    SparseMatrix<Number> A = SparseMatrixFactory::bandMatrix<Number>(512, band);
    std::cout << A.ssoraInverse(1.0).toString() << "\n";
    
    SparseMatrix<Number> mat = SparseMatrixFactory::triDiagonal<Number>(1, -1, 2, -1);
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";

    mat = SparseMatrixFactory::triDiagonal<Number>(2, -1, 2, -1);
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";
    
    mat = SparseMatrixFactory::triDiagonal<Number>(3, -1, 2, -1);
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";

    mat = std::move(SparseMatrixFactory::triDiagonal<Number>(6, -1, 2, -1));
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";
    return 0;
}


     
