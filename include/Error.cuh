#ifndef ERROR_CUH
#define ERROR_CUH

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>

void __checkCuda(cudaError_t error, const char* file, int line) {
    if(error != cudaSuccess) {
	std::stringstream ss;
	ss << file << "(" << line << ")";
	std::string file_and_line;
	ss >> file_and_line;
	throw thrust::system_error(error, thrust::cuda_category(), file_and_line);
    }
}
#define checkCuda(ans) { __checkCuda((ans), __FILE__, __LINE__); }
    
#endif