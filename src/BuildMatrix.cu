#include <iostream>
#include "SparseMatrix.hpp"

int main() {
    int dim = 16;
    
    auto mat = SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>>::triDiagonal(dim, -1, 2, -1);
    std::cout << mat.toString() << "\n";

    
    std::vector<float> band{5,4,3,2,1};
    mat = SparseMatrix<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>>::bandMatrix(dim, band);
    std::cout << mat.toString() << "\n";
    
}