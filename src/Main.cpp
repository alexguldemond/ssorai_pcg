#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include "LinearSolve.hpp"

#include <iostream>
#include <cstdlib>

typedef double Number;

int main(int argc, char** argv) {
    int dim = 1024;
    if (argc > 1) {
	dim = atoi(argv[1]);
    }
    double relax = 1;
    if (argc > 2) {
	relax = atof(argv[2]);
    }

    std::cout << "Solving with dim = " << dim << ", relax = " << relax << "\n";
    SparseMatrix<Number> A = SparseMatrix<Number>::triDiagonal(dim, -1, 2,-1);
    std::vector<Number> bVec(dim , 1);
    DenseVector<Number> b(bVec);

    SsoraPcgSolver<Number> solver(A, b, .1, relax, dim);
    Result<Number> result = solver.solve();
    
    std::cout << "Iterations: " << result.iterations << "\n"; 
}
