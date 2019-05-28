#ifndef GPU_SOLVE_CUH
#define GPU_SOLVE_CUH

#include "DenseVector.hpp"
#include "SparseMatrix.hpp"
#include "LinearSolve.hpp"
#include "Parallel.cuh"
#include "DevicePtr.cuh"

template <class T>
class GpuSolver: LinearSolver<T> {
private:
    const SparseMatrix<T>& mat;
    const DenseVector<T>& vec;
    double threshold;
    double relaxation;
    int maxIter;
    int threadsPerBlock;
    int dim;
    int nnz;


    void matVec(const DevicePtr<T>& x,
		const DevicePtr<T>& a_entries,
		const DevicePtr<T>& a_cols,
		const DevicePtr<T>& a_rowPtrs) const;

    void aXPlusY(const T scalar, const DevicePtr<T> x, const DevicePtr<T> y) const;
    
    T dotProd(const DevicePtr<T> vec1, const DevicePtr<T> vec2) const ;
    
    void initResidual(const DevicePtr<T>& x,
		 const DevicePtr<T>& a_entries,
		 const DevicePtr<T>& a_cols,
		 const DevicePtr<T>& a_rowPtrs,
		 DevicePtr<T>& residual) const;

			     
public:
    GpuSolver(const SparseMatrix<T>& matrix,
	      const DenseVector<T>& vector,
	      double thresh,
	      double relax,
	      int iters,
	      int threads):
	mat(matrix),
	vec(vector),
	threshold(thresh),
	relaxation(relax),
	maxIter(iters),
	threadsPerBlock(threads),
	dim(matrix.dim()),
	nnz(matrix.nonZeroEntries()) { }
    
    virtual Result<T> solve() const override;
    
};

template <class T>
void GpuSolver<T>::matVec(const DevicePtr<T>& a_entries,const DevicePtr<T>& a_cols,
			  const DevicePtr<T>& a_rowPtrs, const DevicePtr<T>& x,
			  DevicePtr<T>& result) const {

    int gridSize = (nnz / threadsPerBlock) + 1;
    int sharedSize = nnz * sizeof(T);
    sparseMatrixVectorProduct<<<gridSize,threadsPerBlock,size>>>(a_entries.raw(), a_cols.raw(), a_rowPtrs.raw(),
								 x.raw(), &dim, result.raw());
}

template <class T>
void GpuSolver<T>::aXPlusY(const T scalar, const DevicePtr<T> x, const DevicePtr<T> y) const {
    DevicePtr<T> deviceScalar(scalar);
}

template <class T>
void GpuSolver<T>::initResidual(const DevicePtr<T>& x,
				const DevicePtr<T>& a_entries,
				const DevicePtr<T>& a_cols,
				const DevicePtr<T>& a_rowPtrs,
				const Device<Ptr>T& b,
				DevicePtr<T>& residual) const {

    matVec(a_entries, a_cols, a_rowPtrs, x, residual);
    float scalar = -1;
    int gridSize = dim / threadsPerBlock;
    int size = threadsPerBlock * sizeof(T);
    aXPlusY<<<gridSize, threadsPerBlock, size>>>(&scalar, residual.raw(), b.raw, &dim, residual.raw());
}

Result<T> GpuSolver<T>::solve() const {
    DenseVector<T> x = DenseVector<T>::zero(dim);
    SparseMatrix<T> p = mat.ssoraInverse(relaxation);

    //Setup all device data 
    DevicePtr<T> device_x(x.data(), dim);
    
    DevicePtr<T> device_p_entries(&p.entries[0], nnz);
    DevicePtr<T> device_p_cols(&p.entries[0], nnz);
    DevicePtr<T> device_p_rowPtrs(&p.entries[0], dim + 1);

    DevicePtr<T> device_a_entries(&mat.entries[0], nnz);
    DevicePtr<T> device_a_cols(&mat.entries[0], nnz);
    DevicePtr<T> device_a_rowPtrs(&mat.entries[0], dim + 1);

    DevicePtr<T> device_b(vec.data(), dim);

    DevicePtr<T> device_r(dim);
    initResidual(device_x, device_a_entries, device_a_cols,
		 device_a_rowPtrs, device_b, dim, nnz, device_r);
    DevicePtr<T> device_next_r(dim);

    DevicePtr<T> device_z(dim);
    matVec(device_p_entries, device_p_cols, device_p_entries, device_r, device_z);

    DevicePtr<T> devie_next_z(dim);
    
    
}

#endif