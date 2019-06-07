#ifndef GPU_SOLVE_CUH
#define GPU_SOLVE_CUH

#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "LinearSolve.hpp"
#include "Parallel.cuh"
#include "gpu_memory.cuh"

#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>
#include <algorithm>


#define div_up(a, b) ((a/b) + ((a % b) == 0))

template <class T>
using DeviceVector = DenseVector<T, gpu::CudaDeleter<T[]> >;

template <class T, class Deleter = std::default_delete<T[]>, class IntDeleter = std::default_delete<int[]> >
class GpuSolver: LinearSolver<T, Deleter> {
private:
    SparseMatrix<T, Deleter, IntDeleter>& mat;
    DenseVector<T, Deleter>& vec;
    double threshold;
    double relaxation;
    int maxIter;
    int threadsPerBlock;

    void matVec(const gpu::device_ptr<T[]>& a_entries,
		const gpu::device_ptr<int[]>& a_cols,
		const gpu::device_ptr<int[]>& a_rowPtrs,
		const DeviceVector<T>& x,
		DeviceVector<T>& result) const;

    void aXPlusY(T scalar, const DeviceVector<T>& x, const DeviceVector<T>& y, DeviceVector<T>& result) const;
    
    void dotProduct(const DeviceVector<T>& vec1, const DeviceVector<T>& vec2, gpu::device_ptr<T>& result) const;

    void copyVector(const DeviceVector<T>& src, DeviceVector<T>& dest) const;
    
    int dim() const {
	return mat.dim();
    }

    int nnz() const {
	return mat.nonZeroEntries();
    }
			     
public:
    GpuSolver(SparseMatrix<T, Deleter, IntDeleter>& matrix,
	      DenseVector<T, Deleter>& vector,
	      double thresh,
	      double relax,
	      int iters,
	      int threads):
	mat(matrix),
	vec(vector),
	threshold(thresh),
	relaxation(relax),
	maxIter(iters),
	threadsPerBlock(threads) { }
    
    virtual Result<T> solve() const override;
    
};

template <class T, class Deleter, class IntDeleter>
void GpuSolver<T, Deleter, IntDeleter>::matVec(const gpu::device_ptr<T[]>& a_entries,
					       const gpu::device_ptr<int[]>& a_cols,
					       const gpu::device_ptr<int[]>& a_rowPtrs,
					       const DeviceVector<T>& x,
					       DeviceVector<T>& result) const {
    kernel::sparseMatrixVectorProduct<<<div_up(dim(),threadsPerBlock),threadsPerBlock>>>(a_entries.get(), a_cols.get(), a_rowPtrs.get(),
											x.entries.get(), dim(), result.entries.get());
    checkCuda(cudaPeekAtLastError());
}

template <class T, class Deleter, class IntDeleter>
void GpuSolver<T, Deleter, IntDeleter>::aXPlusY(T scalar,
						const DeviceVector<T>& x,
						const DeviceVector<T>& y,
						DeviceVector<T>& result) const {
    
    kernel::aXPlusY<<< div_up(dim(),threadsPerBlock), threadsPerBlock>>>(scalar,
									 x.entries.get(),
									 y.entries.get(),
									 dim(),
									 result.entries.get());
    checkCuda(cudaPeekAtLastError());
}

template <class T, class Deleter, class IntDeleter>
void GpuSolver<T, Deleter, IntDeleter>::dotProduct(const DeviceVector<T>& vec1,
						   const DeviceVector<T>& vec2,
						   gpu::device_ptr<T>& result) const {
    checkCuda(cudaMemset(result.get(), 0, sizeof(T)));
    int blocks = div_up(dim(),threadsPerBlock);
    kernel::dotProduct<<<blocks, threadsPerBlock, threadsPerBlock*sizeof(T)>>>(vec1.entries.get(), vec2.entries.get(), dim(), result.get());
    checkCuda(cudaPeekAtLastError());
}

template <class T, class Deleter, class IntDeleter>
void GpuSolver<T, Deleter, IntDeleter>::copyVector(const DeviceVector<T>& src, DeviceVector<T>& dest) const {
    kernel::copyArray<<< div_up(dim(),threadsPerBlock), threadsPerBlock>>>(src.entries.get(), dest.entries.get(), dim());
    checkCuda(cudaPeekAtLastError());
}

template <class T, class Deleter, class IntDeleter>
Result<T> GpuSolver<T, Deleter, IntDeleter>::solve() const {
    std::cout << "Solving...\n";
    DenseVector<T> x = DenseVectorFactory::zero<T>(dim());

    std::cout << "Computing ssora inverse...\n";
    boost::posix_time::ptime time_start(boost::posix_time::microsec_clock::local_time());
    auto p_entries = mat.ssoraInverseEntries(relaxation);
    boost::posix_time::ptime time_end(boost::posix_time::microsec_clock::local_time());
    boost::posix_time::time_duration ssoraDuration(time_end - time_start);

    time_start = boost::posix_time::microsec_clock::local_time();
    //Setup all device data
    std::cout << "Setting up device data...\n";

    DeviceVector<T> device_x(dim(), gpu::make_device<T>(x.entries, dim()));
    
    auto device_a_entries = gpu::make_device<T, Deleter>(mat.entries, nnz());
    auto device_a_cols = gpu::make_device<int, IntDeleter>(mat.cols, nnz());
    auto device_a_rowPtrs = gpu::make_device<int, IntDeleter>(mat.rowPtrs, dim() + 1);

    auto device_p_entries = gpu::make_device<T, Deleter>(p_entries, nnz());
    
    //r = b - a.x = b
    DeviceVector<T> device_r(dim(), gpu::make_device<T, Deleter>(vec.entries, dim()));
    DeviceVector<T> device_next_r(dim(), gpu::make_device<T>( dim() ));

    //z = p.r
    DeviceVector<T> device_z(dim(), gpu::make_device<T>( dim() ));
    matVec(device_p_entries, device_a_cols, device_a_rowPtrs ,device_r, device_z);
    DeviceVector<T> device_next_z(dim(), gpu::make_device<T>( dim() ));

    //cache r.z
    gpu::device_ptr<T> device_r_dot_z = gpu::make_device<T>();
    dotProduct(device_r, device_z, device_r_dot_z);
    T r_dot_z = gpu::get_from_device<T>(device_r_dot_z);

    //d = z
    DeviceVector<T> device_d(dim(), gpu::make_device<T>( dim() ));
    copyVector(device_z, device_d);

    //a_d = a.d
    DeviceVector<T> device_a_d(dim(), gpu::make_device<T>( dim() ));
    matVec(device_a_entries, device_a_cols, device_a_rowPtrs, device_d, device_a_d);

    //cache r.r
    gpu::device_ptr<T> device_r_dot_r = gpu::make_device<T>();
    dotProduct(device_r, device_r, device_r_dot_r);

    //cache d.a.d
    gpu::device_ptr<T> device_d_a_d = gpu::make_device<T>();
    dotProduct(device_d, device_a_d, device_d_a_d);

    gpu::device_ptr<T> device_next_r_dot_z = gpu::make_device<T>();
    int count = 0;

    //Begin algorithm
    T r_dot_r;
    while (( r_dot_r = gpu::get_from_device<T>(device_r_dot_r)) > threshold && count < maxIter) {

	if (count % 100 == 0) {
	    boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
	    std::string time = boost::posix_time::to_simple_string(timeLocal);
	    std::cout << time << " k = " << count << ", r.r = " << r_dot_r << "\n";
	}
	
	//Compute alpha = r.z/d.a.d
	T stepSize = r_dot_z / gpu::get_from_device<T>(device_d_a_d);

	//x = x + alpha*d
	aXPlusY(stepSize, device_d, device_x, device_x);
	
	//r = r - alpha*a.d
	aXPlusY(-stepSize, device_a_d, device_r, device_next_r);
	
	//z = p.r
	matVec(device_p_entries, device_a_cols, device_a_rowPtrs, device_next_r, device_next_z);
	
	//next_r.z = r.z
	dotProduct(device_next_r, device_next_z, device_next_r_dot_z);
	T next_r_dot_z = gpu::get_from_device<T>(device_next_r_dot_z);

	//beta = rk+1 . zk+1 / rk . zk
	T update = next_r_dot_z / r_dot_z;
	r_dot_z = next_r_dot_z;
	
	//d = z + beta*d
	aXPlusY(update, device_d, device_next_z, device_d);
	
	//a_d = a.d
	matVec(device_a_entries, device_a_cols, device_a_rowPtrs, device_d, device_a_d);
	
	//Update
	device_r = std::move(device_next_r);
	device_z = std::move(device_next_z);
	dotProduct(device_d, device_a_d, device_d_a_d);
	dotProduct(device_r, device_r,device_r_dot_r);
	count++;
    }
    
    DenseVector<T> result(dim(), gpu::get_from_device<T>(device_x.entries, dim()));
    time_end = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration pcgDuration(time_end - time_start);
    return Result<T>(result, count, r_dot_r, ssoraDuration , pcgDuration);
    
    
}

#endif