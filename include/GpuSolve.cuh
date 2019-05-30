#ifndef GPU_SOLVE_CUH
#define GPU_SOLVE_CUH

#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "LinearSolve.hpp"
#include "Parallel.cuh"
#include "DevicePtr.cuh"

#include "boost/date_time/posix_time/posix_time.hpp"
#include <iostream>
#include <algorithm>


#define max(a, b) a >= b ? a : b

template <class T>
struct DeviceMatrix {
    DevicePtr<T> entries;
    DevicePtr<int> cols;
    DevicePtr<int> rowPtrs;

    DeviceMatrix(const SparseMatrix<T>& mat):
	entries(&mat.entries[0], mat.nonZeroEntries()),
	cols(&mat.cols[0], mat.nonZeroEntries()),
	rowPtrs(&mat.rowPtrs[0], mat.dim() + 1) { }

    DeviceMatrix(const DeviceMatrix<T>& other) = delete;
    DeviceMatrix(DeviceMatrix<T>&& other) = delete;
    DeviceMatrix<T>& operator=(const DeviceMatrix<T>& other) = delete;
    DeviceMatrix<T>& operator=(DeviceMatrix<T>&& other) = delete;
};

template <class T>
class GpuSolver: LinearSolver<T> {
private:
    SparseMatrix<T>& mat;
    DenseVector<T>& vec;
    double threshold;
    double relaxation;
    int maxIter;
    int threadsPerBlock;

    void matVec(const DeviceMatrix<T>& a, const DevicePtr<T>& x, DevicePtr<T>& result) const;

    void aXPlusY(T scalar, const DevicePtr<T>& x, const DevicePtr<T>& y, DevicePtr<T>& result) const;
    
    void dotProduct(const DevicePtr<T>& vec1, const DevicePtr<T>& vec2, DevicePtr<T>& result) const;

    void copyVector(const DevicePtr<T>& src, DevicePtr<T>& dest) const;
    
    int dim() const {
	return mat.dim();
    }

    int nnz() const {
	return mat.nonZeroEntries();
    }
			     
public:
    GpuSolver(SparseMatrix<T>& matrix,
	      DenseVector<T>& vector,
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

template <class T>
void GpuSolver<T>::matVec(const DeviceMatrix<T>& a, const DevicePtr<T>& x,  DevicePtr<T>& result) const {
    int nonZero = nnz();
    int gridSize = (nonZero/threadsPerBlock) + (nonZero % threadsPerBlock != 0);
    int sharedSize = nonZero * sizeof(T);
    kernel::sparseMatrixVectorProduct<<<gridSize,threadsPerBlock, sharedSize>>>(a.entries.raw(), a.cols.raw(), a.rowPtrs.raw(),
										x.raw(), dim(), result.raw());
    checkCuda(cudaPeekAtLastError());
}

template <class T>
void GpuSolver<T>::aXPlusY(T scalar, const DevicePtr<T>& x,
			   const DevicePtr<T>& y, DevicePtr<T>& result) const {

    kernel::aXPlusY<<< max(dim()/threadsPerBlock,1), threadsPerBlock>>>(scalar, x.raw(), y.raw(), dim(), result.raw());
    checkCuda(cudaPeekAtLastError());
}

template <class T>
void GpuSolver<T>::dotProduct(const DevicePtr<T>& vec1, const DevicePtr<T>& vec2, DevicePtr<T>& result) const {
    int blocks = max(dim()/threadsPerBlock, 1);
    kernel::dotProduct<<<blocks, threadsPerBlock, threadsPerBlock*sizeof(T) >>>(vec1.raw(), vec2.raw(), dim(), result.raw());
}

template <class T>
void GpuSolver<T>::copyVector(const DevicePtr<T>& src, DevicePtr<T>& dest) const {
    kernel::copyArray<<< max(dim()/threadsPerBlock, 1), threadsPerBlock>>>(src.raw(), dest.raw(), dim());
    checkCuda(cudaPeekAtLastError());
}

template <class T>
Result<T> GpuSolver<T>::solve() const {
    DenseVector<T> x = DenseVector<T>::zero(dim());
    SparseMatrix<T> p = mat.ssoraInverse(relaxation);

    //Setup all device data 
    DevicePtr<T> device_x(x.data(), dim());

    DeviceMatrix<T> device_p(p);
    DeviceMatrix<T> device_a(mat);

    DevicePtr<T> device_b(vec.data(), dim());

    //r = b - a.x = b
    DevicePtr<T> device_r(dim());
    copyVector(device_b, device_r);
    DevicePtr<T> device_next_r(dim());

    //z = p.r
    DevicePtr<T> device_z(dim());
    matVec(device_p, device_r, device_z);
    DevicePtr<T> device_next_z(dim());

    //cache r.z
    DevicePtr<T> device_r_dot_z;
    dotProduct(device_r, device_z, device_r_dot_z);
    T r_dot_z = device_r_dot_z.get();

    //d = z
    DevicePtr<T> device_d(dim());
    copyVector(device_z, device_d);

    //a_d = a.d
    DevicePtr<T> device_a_d(dim());
    matVec(device_a, device_d, device_a_d);

    //cache r.r
    DevicePtr<T> device_r_dot_r;
    dotProduct(device_r, device_r, device_r_dot_r);

    //cache d.a.d
    DevicePtr<T> device_d_a_d;
    dotProduct(device_d, device_a_d, device_d_a_d);
    
    DevicePtr<T> device_next_r_dot_z;
    
    int count = 0;

    //Begin algorithm
    T r_dot_r;
    while (( r_dot_r = device_r_dot_r.get()) > threshold && count < maxIter) {

	boost::posix_time::ptime timeLocal = boost::posix_time::second_clock::local_time();
	std::string time = boost::posix_time::to_simple_string(timeLocal);
	std::cout << time << " k = " << count << ", r.r = " << r_dot_r << "\n";

	//Compute alpha = r.z/d.a.d
	T stepSize = r_dot_z / device_d_a_d.get();

	//x = x + alpha*d
	aXPlusY(stepSize, device_d, device_x, device_x);
	
	//r = r - alpha*a.d
	aXPlusY(-stepSize, device_a_d, device_r, device_next_r);
	
	//z = p.r
	matVec(device_p, device_next_r, device_next_z);
	
	//next_r.z = r.z
	dotProduct(device_next_r, device_next_z, device_next_r_dot_z);
	T next_r_dot_z = device_next_r_dot_z.get();

	//beta = rk+1 . zk+1 / rk . zk
	T update = next_r_dot_z / r_dot_z;
	r_dot_z = next_r_dot_z;
	
	//d = z + beta*d
	aXPlusY(update, device_d, device_next_z, device_d);
	
	//a_d = a.d
	DenseVector<T> host_d(device_d.getAll());
	matVec(device_a, device_d, device_a_d);
	
	//Update
	device_r = std::move(device_next_r);
	device_z = std::move(device_next_z);
	dotProduct(device_d, device_a_d, device_d_a_d);
	dotProduct(device_r, device_r,device_r_dot_r);
	count++;
    }

    DenseVector<T> result(device_x.getAll());
    return Result<T>(result, count);
    
    
}

#endif