#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include <iostream>
#include <vector>

typedef double Number;

int main() {
    
    std::vector<double> entries{6, -1, -1, 7, -2, -2, 8, -3, -3, 9, -4, -4, 10};
    std::vector<int> cols      {0,  1,  0, 1,  2,  1, 2,  3,  2, 3,  4,  3,  4};
    std::vector<int> rowPtrs{0, 2, 5, 8, 11, 13};
    
    SparseMatrix<Number> mat(entries, cols, rowPtrs);
    
    std::cout << mat.toString() << "\n";
    std::cout << mat.ssoraInverse(1.0).toString() << "\n";
    return 0;
}


     
