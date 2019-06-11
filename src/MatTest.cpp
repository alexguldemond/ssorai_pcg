#include "SparseMatrix.hpp"
#include <iostream>

int main() {
    std::vector<float> vec{100,1,1,1,1,1,5,1,1};
    SparseMatrix<float> mat = SparseMatrix<float>::bandMatrix(32, vec);
    std::cout << mat.toString() << "\n";

    auto diag = mat.getAllDiagonals();
    for (int i = 0; i < 32; i++) {
	std::cout << "[i,i] = " << diag[i] << "\n";
    }
    
}
