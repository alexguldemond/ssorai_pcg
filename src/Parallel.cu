#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include "GpuSolve.cuh"
#include "gpu_memory.cuh"

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

    auto A = SparseMatrixFactory::triDiagonal<float, gpu::CudaHostDeleter<float[]>, gpu::CudaHostDeleter<int[]>>(dim, -1, 2, -1,
														 gpu::CudaHostDeleter<float[]>(),
														 gpu::CudaHostDeleter<int[]>(),
														 gpu::safe_host_malloc<float>,
														 gpu::safe_host_malloc<int>);
    
    auto b = DenseVectorFactory::constant<float, gpu::CudaHostDeleter<float[]>>(dim, 1,
										gpu::CudaHostDeleter<float[]>(),
										gpu::safe_host_malloc<float>);
										
    
    GpuSolver<float, gpu::CudaHostDeleter<float[]>, gpu::CudaHostDeleter<int[]>> solver(A, b, 50000, relax, dim, threadsPerBlock);
    auto result = solver.solve();

    std::cout << "Iterations: " << result.iterations << "\n";
    std::cout << "Final residual: " << result.residualNormSquared << "\n";
    std::cout << "Ssora compute time: " << result.ssoraDuration << "\n";
    std::cout << "Pcg comput time: " << result.pcgDuration << "\n";

}
