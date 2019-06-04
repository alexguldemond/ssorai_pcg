#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include "GpuSolve.cuh"

#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    int dim = 1024;
    if (argc > 1) {
	dim = atoi(argv[1]);
    }
    double relax = 1;
    if (argc > 2) {
	relax = atof(argv[2]);
    }
    int threadsPerBlock = 1024;
    if (argc > 3) {
	threadsPerBlock = atoi(argv[3]);
    }

    std::cout << "Solving with dim = " << dim << ", relax = " << relax << "\n";
    SparseMatrix<float> A = SparseMatrix<float>::triDiagonal(dim, -1, 2,-1);
    std::vector<float> bVec(dim , 1);
    DenseVector<float> b(bVec);
    
    GpuSolver<float> solver(A, b, .1, relax, dim, threadsPerBlock);
    Result<float> result = solver.solve();

    //std::cout << result.result.toString() << "\n";
    std::cout << "Iterations: " << result.iterations << "\n"; 
}
