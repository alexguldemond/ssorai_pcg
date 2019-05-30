#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include "LinearSolve.hpp"
#include <iostream>
#include <vector>
#include <utility>

typedef double Number;

int main() {

    int dim = 32;
    SparseMatrix<Number> A = SparseMatrix<Number>::triDiagonal(dim, -1, 2,-1);
    std::vector<Number> bVec(dim , 1);
    DenseVector<Number> b(bVec);

    std::cout << "Solveing the following:\n";
    std::cout << A.toString() << "\n";
    std::cout << b.toString() << "\n";

    SsoraPcgSolver<Number> solver(A, b, 0.01, 1, 100);
    Result<Number> result = solver.solve();
    
    std::cout << result.result.toString() << "\n";
    std::cout << "Iterations: " << result.iterations << "\n"; 
}
