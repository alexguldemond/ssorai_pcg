#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include "LinearSolve.hpp"

#include <iostream>
#include <cstdlib>

typedef float Number;

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

    SparseMatrix<Number> A = SparseMatrixFactory::triDiagonal<Number>(dim, -1,2,-1);
    DenseVector<Number> b = DenseVectorFactory::constant<Number>(dim, 1);
    
    SsoraPcgSolver<Number> solver(A, b, 50000, relax, dim);
    Result<Number> result = solver.solve();
    
    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Result Residual: " << result.residualNormSquared << "\n";
    std::cout << "Ssora compute time: " << result.ssoraDuration << "\n";
    std::cout << "Pcg comput time: " << result.pcgDuration << "\n";
}
