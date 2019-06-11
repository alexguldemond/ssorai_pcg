#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include "GpuSolve.cuh"
#include "gpu_memory.cuh"

#include <iostream>
#include <cstdlib>
#include "boost/program_options.hpp"

using namespace boost::program_options;

int main(int argc, char** argv) {
    try {
	options_description desc{"Options"};
	desc.add_options()
	    ("help,h", "Help screen")
	    ("dim,d", value<int>()->default_value(1024), "dimension")
	    ("relax,r", value<float>()->default_value(1), "relaxation")
	    ("threshold,t", value<float>()->default_value(.1), "termination threshold")
	    ("threadsPerBlock, p", value<int>()->default_value(1024), "Threads Per Block");
	
	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);
	notify(vm);
	
	if (vm.count("help")) {
	    std::cout << desc << '\n';
	    return 0;
	}
	
	int dim = vm["dim"].as<int>();
	float relax = vm["relax"].as<float>();
	float threshold = vm["threshold"].as<float>();
	int threadsPerBlock = vm["threadsPerBlock"].as<int>();
	
	std::cout << "Solving with dim = " << dim << ", relax = " << relax << ", threshold = " << threshold << "\n";

	std::vector<float> band(100, 1);
	for (std::size_t i = 0; i < band.size(); i++) {
	    band[i] = i;
	}
	band[0] = 100 * 101;
	auto A = SparseMatrix<float, gpu::CudaHostDeleter<float[]>, gpu::CudaHostDeleter<int[]>>::bandMatrix(dim, band);
	
	auto b = DenseVector<float, gpu::CudaHostDeleter<float[]>>::constant(dim, 1);
	
	
	GpuSolver<float, gpu::CudaHostDeleter<float[]>, gpu::CudaHostDeleter<int[]>> solver(A, b, threshold, relax, dim, threadsPerBlock);
	auto result = solver.solve();
	
	std::cout << "\nIterations: " << result.iterations << "\n";
	std::cout << "Final residual: " << result.residualNormSquared << "\n";
	std::cout << "Ssora compute time: " << result.ssoraDuration << "\n";
	std::cout << "Pcg comput time: " << result.pcgDuration << "\n";
    } catch (const error &ex) {
	std::cerr << ex.what() << '\n';
    }    
}
