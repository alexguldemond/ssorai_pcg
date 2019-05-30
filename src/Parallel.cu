#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include "GpuSolve.cuh"


int main() {

    int dim = 32;
    SparseMatrix<float> A = SparseMatrix<float>::triDiagonal(dim, -1, 2,-1);
    std::vector<float> bVec(dim , 1);
    DenseVector<float> b(bVec);

    std::cout << "Solveing the following:\n";
    std::cout << A.toString() << "\n";
    std::cout << b.toString() << "\n";
    
    GpuSolver<float> solver(A, b, 0.01, 1, 1024, 1024);
    Result<float> result = solver.solve();

    std::cout << result.result.toString() << "\n";
    std::cout << "Iterations: " << result.iterations << "\n"; 
}
