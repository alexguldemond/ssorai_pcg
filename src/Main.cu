#include "SparseMatrix.hpp"
#include "DenseVector.hpp"
#include "LinearSolve.hpp"
#include "gpu_memory.cuh"

#include <iostream>
#include <cstdlib>
#include "boost/program_options.hpp"

using namespace boost::program_options;

template<class T, class Deleter, class IntDeleter, class ScalarDeleter>
void run(int dim, T relax, T threshold) {
    SparseMatrix<float, Deleter, IntDeleter> A = SparseMatrix<float, Deleter, IntDeleter>::triDiagonal(dim, -1, 2, -1);
    DenseVector<float, Deleter> b = DenseVector<float, Deleter>::constant(dim, 1);
    
    SsoraPcgSolver<float, Deleter, IntDeleter, ScalarDeleter> solver(A, b, threshold, relax, dim);
    Result<T, Deleter> result = solver.solve();

    std::cout << "\nIterations: " << result.iterations << "\n";
    std::cout << "Result Residual: " << result.residualNormSquared << "\n";
    std::cout << "Ssora compute time: " << result.ssoraDuration << "\n";
    std::cout << "Pcg compute time: " << result.pcgDuration << "\n";
}

int main(int argc, char** argv) {
    try {
	bool gpuMode = false;
	options_description desc{"Options"};
	desc.add_options()
	    ("help,h", "Help screen")
	    ("dim,d", value<int>()->default_value(1024), "dimension")
	    ("relax,r", value<float>()->default_value(1), "relaxation")
	    ("threshold,t", value<float>()->default_value(.1), "termination threshold")
	    ("gpu,g", bool_switch(&gpuMode), "gpu mode")
	    ("threadsPerBlock,p", value<int>()->default_value(1024), "Threads Per Block, only matters in gpu mode")
	    ("blocks,b", value<int>()->default_value(0), "Number of blocks, only matters in gpu mode. Default is dim / threadsPerBlock rounded up");
	
	
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
	int blocks = vm["blocks"].as<int>() == 0 ? kernel::roundUpDiv(dim, threadsPerBlock) : vm["blocks"].as<int>();
	std::string mode = gpuMode ? "gpu" : "cpu";
	
	std::cout << "Solving with dim = " << dim << ", relax = " << relax << ", threshold = " << threshold << ", mode = " << mode <<"\n";
	if (gpuMode) {
	    LinearAlgebra::threadsPerBlock = threadsPerBlock;
	    LinearAlgebra::blocks = blocks;
	    std::cout << "Threads Per Block = " << LinearAlgebra::threadsPerBlock << "\n";
	    std::cout << "Num Blocks = " << LinearAlgebra::blocks << "\n";
	    run<float, gpu::CudaDeleter<float[]>, gpu::CudaDeleter<int[]>, gpu::CudaDeleter<float>>(dim, relax, threshold);
	} else {
	    run<float, std::default_delete<float[]>, std::default_delete<int[]>, std::default_delete<float>>(dim, relax, threshold);
	}
    } catch (const error &ex) {
	std::cerr << ex.what() << '\n';
    }
}
