#ifndef DENSEVECTOR_HPP
#define DENSEVECTOR_HPP

#include <memory>
#include <utility>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>

#include "allocater.hpp"

//Declare lazy operators
template <class T, class Deleter>
class PlusAx;

template <class T, class Deleter, class IntDeleter>
class MatVec;

template <class T, class Deleter = std::default_delete<T[]> >
class DenseVector {
private:
    int _dim;
    std::unique_ptr<T[], Deleter> _entries;
public:
    static DenseVector<T, Deleter> incremental(int dim, T startValue);

    static DenseVector<T, Deleter> constant(int dim, T value);

    static DenseVector<T, Deleter> zero(int dim);
    
    DenseVector(int dim, const std::unique_ptr<T[], Deleter>& newEntries);

    DenseVector(int dim, std::unique_ptr<T[], Deleter>&& entries);

    DenseVector(const DenseVector<T, Deleter>& other);

    DenseVector(DenseVector<T, Deleter>&& other);
    
    DenseVector<T, Deleter>& operator=(const DenseVector<T, Deleter>& other);

    DenseVector<T, Deleter>& operator=(DenseVector<T, Deleter>&& other);

    const std::unique_ptr<T[], Deleter>& entries() const {
	return _entries;
    }

    std::unique_ptr<T[], Deleter>& entries() {
	return _entries;
    }
    
    int dim() const;

    template<class ScalarDeleter>
    void dot(const DenseVector<T, Deleter>& other, std::unique_ptr<T, ScalarDeleter>& result) const;
    
    T dot(const DenseVector<T, Deleter>& other) const;

    T normSquared() const;

    std::string toString() const;

    const T& operator[](int i) const;
    
    T& operator[](int i);

    DenseVector<T, Deleter>& updateAx(const T& scalar, const DenseVector<T, Deleter>& other);

    DenseVector(const PlusAx<T, Deleter>& op);
    
    DenseVector<T, Deleter>& operator=(const PlusAx<T, Deleter>&& op);

    void plusAx(const T& scalar, const DenseVector<T, Deleter>& other, DenseVector<T, Deleter>& result) const;
    
    PlusAx<T, Deleter> plusAx(const T& scalar, const DenseVector<T, Deleter>& other) const;

    template <class IntDeleter>
    DenseVector(const MatVec<T, Deleter, IntDeleter>& op);

    template <class IntDeleter>
    DenseVector<T, Deleter>& operator=(const MatVec<T, Deleter, IntDeleter>& op);
};

template <class T, class Deleter>
DenseVector<T, Deleter> DenseVector<T, Deleter>::incremental(int dim, T startValue) {
	T* entries = Allocater<T, Deleter>::allocate(dim);
	std::iota(entries, entries+dim, startValue);
	return DenseVector<T, Deleter>(dim, std::unique_ptr<T[], Deleter>(entries, Deleter()));
}

template <class T, class Deleter>
DenseVector<T, Deleter> DenseVector<T, Deleter>::constant(int dim, T value) {
	T* entries = Allocater<T, Deleter>::allocate(dim);
	std::fill(entries, entries+dim, value);
	return DenseVector<T, Deleter>(dim, std::unique_ptr<T[], Deleter>(entries, Deleter()));
}

template <class T, class Deleter>
DenseVector<T, Deleter> DenseVector<T, Deleter>::zero(int dim) {
    return constant(dim, 0);
}

template <class T, class Deleter>
DenseVector<T, Deleter>::DenseVector(int dim,
				     const std::unique_ptr<T[], Deleter>& newEntries): _dim(dim)  {
    T* vals = Allocater<T, Deleter>::allocate(dim);
    std::copy(newEntries.get(), newEntries.get() + _dim, vals);
    _entries = std::unique_ptr<T[], Deleter>(vals, Deleter());
}

template <class T, class Deleter>
DenseVector<T, Deleter>::DenseVector(int dim, std::unique_ptr<T[], Deleter>&& _entries): _dim(dim),
											 _entries(std::move(_entries)) {
    //Intentionally empty
}

template <class T, class Deleter>
DenseVector<T, Deleter>::DenseVector(const DenseVector<T, Deleter>& other): _dim(other._dim) {
    T* newEntries = Allocater<T, Deleter>::allocate(_dim);
    std::copy(other._entries.get(), other._entries.get() + _dim, newEntries);
    _entries = std::unique_ptr<T[], Deleter>(newEntries, Deleter());
}

template <class T, class Deleter>
DenseVector<T, Deleter>::DenseVector(DenseVector<T, Deleter>&& other): _dim(other._dim), _entries(std::move(other._entries)) {
    //Intentionally Empty
}

template <class T, class Deleter>
DenseVector<T, Deleter>& DenseVector<T, Deleter>::operator=(const DenseVector<T, Deleter>& other) {
    if (this != &other) {
	if (_dim != other._dim) {
	    throw "Cannot assign to vector of different dimension\n";
	}
	std::copy(_entries, _entries + _dim , other._entries);
    }
    return *this;
}

template <class T, class Deleter>
DenseVector<T, Deleter>& DenseVector<T, Deleter>::operator=(DenseVector<T, Deleter>&& other) {
    if (this != &other) {
	_dim = other._dim;
	_entries.swap(other._entries);
    }
    return *this;
}

template <class T, class Deleter>
inline int DenseVector<T, Deleter>::dim() const  {
    return _dim;
}

template <class T, class Deleter>
template <class ScalarDeleter>
void DenseVector<T, Deleter>::dot(const DenseVector<T, Deleter>& other, std::unique_ptr<T, ScalarDeleter>& result) const {
    *result = 0.0;
    for (int i = 0; i < dim(); i++) {
	*result += this->_entries[i] * other[i];
    }
}

template <class T, class Deleter>
T DenseVector<T, Deleter>::dot(const DenseVector<T, Deleter>& other) const {
    T sum = 0;
    for (int i = 0; i < dim(); i++) {
	sum += _entries[i] * other[i];
    }
    return sum;
}

template <class T, class Deleter>
T DenseVector<T, Deleter>::normSquared() const {
    return this->dot(*this);
}

template <class T, class Deleter>
std::string DenseVector<T, Deleter>::toString() const{
    std::stringstream stream;
    stream << "[ ";
    for (int i = 0; i < dim(); i++) {
	stream << _entries[i] << " ";
    }
    stream << "]";
    return stream.str();
}

template <class T, class Deleter>
const T& DenseVector<T, Deleter>::operator[](int i) const {
    return _entries[i];
}

template <class T, class Deleter>
T& DenseVector<T, Deleter>::operator[](int i) {
    return _entries[i];
}

template <class T, class Deleter>
DenseVector<T, Deleter>& DenseVector<T, Deleter>::updateAx(const T& scalar, const DenseVector<T, Deleter>& other) {
    for (int i = 0; i < dim(); i++) {
	this->_entries[i] += scalar * other[i];
    }
    return *this;
}

//PlusAx laziness
template <class T, class Deleter>
class PlusAx {
private:
    const DenseVector<T, Deleter>& vector1;
    const T& scalar;
    const DenseVector<T, Deleter>& vector2; 

public:
    PlusAx(const DenseVector<T, Deleter>& vec1, const T& scal, const DenseVector<T, Deleter>& vec2): vector1(vec1), scalar(scal),  vector2(vec2) { }

    int dim() const {
	return vector1.dim();
    }
    
    void operator()(DenseVector<T, Deleter>& result) const {
	vector1.plusAx(scalar, vector2, result);
    }
};

template <class T, class Deleter>
DenseVector<T, Deleter>::DenseVector(const PlusAx<T, Deleter>& op): _dim(op.dim()) {
    T* entr = Allocater<T, Deleter>::allocate(op.dim());
    _entries = std::unique_ptr<T[], Deleter>(entr, Deleter());
    op(*this);
}

template <class T, class Deleter>
DenseVector<T, Deleter>& DenseVector<T, Deleter>::operator=(const PlusAx<T, Deleter>&& op) {
    op(*this);
    return *this;
}

template <class T, class Deleter>
void DenseVector<T, Deleter>::plusAx(const T& scalar, const DenseVector<T, Deleter>& other, DenseVector<T, Deleter>& result) const {
    for (int i = 0; i < dim(); i++) {
	result[i] = _entries[i] + scalar*other[i];
    }
}

template <class T, class Deleter>
PlusAx<T, Deleter> DenseVector<T, Deleter>::plusAx(const T& scalar, const DenseVector<T, Deleter>& other) const {
    return PlusAx<T, Deleter>(*this, scalar, other);
}
//end of PlusAx laziness

//CUDA template specialization
#ifdef __CUDACC__
#include "gpu_memory.cuh"
#include "Parallel.cuh"

template <class T>
using DeviceVector = DenseVector<T, gpu::CudaDeleter<T[]> >;

namespace LinearAlgebra {
    int threadsPerBlock = 1024;
}

template <class T>
class DenseVector<T, gpu::CudaDeleter<T[]>> {
private:
    int _dim;
    std::unique_ptr<T[], gpu::CudaDeleter<T[]>> _entries;
public:
    static DenseVector<T, gpu::CudaDeleter<T[]>> incremental(int dim, T startValue);

    static DenseVector<T, gpu::CudaDeleter<T[]>> constant(int dim, T value);

    static DenseVector<T, gpu::CudaDeleter<T[]>> zero(int dim);
    
    DenseVector(int dim, const std::unique_ptr<T[], gpu::CudaDeleter<T[]>>& newEntries);

    DenseVector(int dim, std::unique_ptr<T[], gpu::CudaDeleter<T[]>>&& entries);

    DenseVector(const DenseVector<T, gpu::CudaDeleter<T[]>>& other);

    DenseVector(DenseVector<T, gpu::CudaDeleter<T[]>>&& other);
    
    DenseVector<T, gpu::CudaDeleter<T[]>>& operator=(const DenseVector<T, gpu::CudaDeleter<T[]>>& other);

    DenseVector<T, gpu::CudaDeleter<T[]>>& operator=(DenseVector<T, gpu::CudaDeleter<T[]>>&& other);

    const std::unique_ptr<T[], gpu::CudaDeleter<T[]>>& entries() const {
	return _entries;
    }

    std::unique_ptr<T[], gpu::CudaDeleter<T[]>>& entries() {
	return _entries;
    }
    
    int dim() const;

    void dot(const DenseVector<T, gpu::CudaDeleter<T[]>>& other, std::unique_ptr<T, gpu::CudaDeleter<T>>& result) const;
    
    T dot(const DenseVector<T, gpu::CudaDeleter<T[]>>& other) const;

    T normSquared() const;

    std::string toString() const;

    const T& operator[](int i) const;
    
    T& operator[](int i);

    DenseVector<T, gpu::CudaDeleter<T[]>>& updateAx(const T& scalar, const DenseVector<T, gpu::CudaDeleter<T[]>>& other);

    DenseVector(const PlusAx<T, gpu::CudaDeleter<T[]>>& op);
    
    DenseVector<T, gpu::CudaDeleter<T[]>>& operator=(const PlusAx<T, gpu::CudaDeleter<T[]>>&& op);

    void plusAx(const T& scalar, const DenseVector<T, gpu::CudaDeleter<T[]>>& other, DenseVector<T, gpu::CudaDeleter<T[]>>& result) const;
    
    PlusAx<T, gpu::CudaDeleter<T[]>> plusAx(const T& scalar, const DenseVector<T, gpu::CudaDeleter<T[]>>& other) const;

    DenseVector(const MatVec<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& op);

    DenseVector<T, gpu::CudaDeleter<T[]>>& operator=(const MatVec<T, gpu::CudaDeleter<T[]>, gpu::CudaDeleter<int[]>>& op);
};

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>> DenseVector<T, gpu::CudaDeleter<T[]>>::incremental(int dim, T startValue) {
    T* entries = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(dim);
    kernel::initIncrementalArray<<<kernel::roundUpDiv(dim, LinearAlgebra::threadsPerBlock), LinearAlgebra::threadsPerBlock>>>(entries, dim, startValue);
    checkCuda(cudaPeekAtLastError());
    return DenseVector<T, gpu::CudaDeleter<T[]>>(dim, gpu::device_ptr<T[]>(entries, gpu::CudaDeleter<T[]>()));
}

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>> DenseVector<T, gpu::CudaDeleter<T[]>>::constant(int dim, T value) {
    T* entries = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(dim);
    kernel::initConstArray<<<kernel::roundUpDiv(dim, LinearAlgebra::threadsPerBlock), LinearAlgebra::threadsPerBlock>>>(entries, dim, value);
    checkCuda(cudaPeekAtLastError());
    return DenseVector<T, gpu::CudaDeleter<T[]>>(dim, gpu::device_ptr<T[]>(entries, gpu::CudaDeleter<T[]>()));
}

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>> DenseVector<T, gpu::CudaDeleter<T[]>>::zero(int dim) {
    T* entries = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(dim);
    checkCuda(cudaMemset(entries, 0, dim * sizeof(T)));
    return DenseVector<T, gpu::CudaDeleter<T[]>>(dim, gpu::device_ptr<T[]>(entries, gpu::CudaDeleter<T[]>()));
}


template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>::DenseVector(const DenseVector<T, gpu::CudaDeleter<T[]>>& other): _dim(other._dim) {
    T* newEntries = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(_dim);
    kernel::copyArray<<<kernel::roundUpDiv(_dim, LinearAlgebra::threadsPerBlock), LinearAlgebra::threadsPerBlock>>>(other._entries.get(), newEntries, _dim);
    checkCuda(cudaPeekAtLastError());
    _entries = gpu::device_ptr<T[]>(newEntries, gpu::CudaDeleter<T[]>());
}

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>::DenseVector(DenseVector<T, gpu::CudaDeleter<T[]>>&& other): _dim(other._dim), _entries(std::move(other._entries)) { }

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>::DenseVector(int dim, std::unique_ptr<T[], gpu::CudaDeleter<T[]>>&& _entries): _dim(dim),
											 _entries(std::move(_entries)) {
    //Intentionally empty
}


template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>& DenseVector<T, gpu::CudaDeleter<T[]>>::operator=(const DenseVector<T, gpu::CudaDeleter<T[]>>& other) {
    if (this != &other) {
	if (_dim != other._dim) {
	    throw "Cannot assign to vector of different dimension\n";
	}
	kernel::copyArray<<<kernel::roundUpDiv(_dim, LinearAlgebra::threadsPerBlock), LinearAlgebra::threadsPerBlock>>>(other._entries.get(), _entries.get(), _dim);
	checkCuda(cudaPeekAtLastError());
    }
    return *this;
}

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>& DenseVector<T, gpu::CudaDeleter<T[]>>::operator=(DenseVector<T, gpu::CudaDeleter<T[]>>&& other) {
    if (this != &other) {
	_dim = other._dim;
	_entries.swap(other._entries);
    }
    return *this;
}

template <class T>
inline int DenseVector<T, gpu::CudaDeleter<T[]>>::dim() const  {
    return _dim;
}

template <class T>
void DenseVector<T, gpu::CudaDeleter<T[]>>::dot(const DenseVector<T, gpu::CudaDeleter<T[]>>& other, std::unique_ptr<T, gpu::CudaDeleter<T>>& result) const {
    checkCuda(cudaMemset(result.get(), 0, sizeof(T)));
    int blocks = kernel::roundUpDiv(dim(),LinearAlgebra::threadsPerBlock);
    kernel::dotProduct<<<blocks,
	LinearAlgebra::threadsPerBlock,
	LinearAlgebra::threadsPerBlock*sizeof(T)>>>(_entries.get(), other.entries().get(), dim(), result.get());
    checkCuda(cudaPeekAtLastError());
}

template <class T>
T DenseVector<T, gpu::CudaDeleter<T[]>>::dot(const DenseVector<T,  gpu::CudaDeleter<T[]>>& other) const {
    gpu::device_ptr<T> result = gpu::make_device<T>();
    dot(other, result);
    return gpu::get_from_device<T>(result);
}

template <class T>
std::string DenseVector<T, gpu::CudaDeleter<T[]>>::toString() const{
    std::unique_ptr<T[]> entries = gpu::get_from_device<T[]>(_entries, _dim);
    DenseVector<T[], gpu::CudaDeleter<T[]>> copy(_dim, entries);
    return copy.toString();
}

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>& DenseVector<T, gpu::CudaDeleter<T[]>>::updateAx(const T& scalar, const DenseVector<T, gpu::CudaDeleter<T[]>>& other) {
    int blocks = kernel::roundUpDiv(dim(),LinearAlgebra::threadsPerBlock);
    kernel::aXPlusY<<<blocks, LinearAlgebra::threadsPerBlock>>>(scalar, other.entries().get(), _entries.get(), dim(),_entries.get());
    return *this;
}

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>::DenseVector(const PlusAx<T, gpu::CudaDeleter<T[]>>& op): _dim(op.dim()) {
    T* entr = Allocater<T, gpu::CudaDeleter<T[]>>::allocate(_dim);
    _entries = std::unique_ptr<T[], gpu::CudaDeleter<T[]>>(entr, gpu::CudaDeleter<T[]>());
    op(*this);
}

template <class T>
DenseVector<T, gpu::CudaDeleter<T[]>>& DenseVector<T, gpu::CudaDeleter<T[]>>::operator=(const PlusAx<T, gpu::CudaDeleter<T[]>>&& op) {
    op(*this);
    return *this;
}

template <class T>
void DenseVector<T, gpu::CudaDeleter<T[]>>::plusAx(const T& scalar,
						   const DenseVector<T, gpu::CudaDeleter<T[]>>& other,
						   DenseVector<T, gpu::CudaDeleter<T[]>>& result) const {
    int blocks = kernel::roundUpDiv(dim(),LinearAlgebra::threadsPerBlock);
    kernel::aXPlusY<<<blocks, LinearAlgebra::threadsPerBlock>>>(scalar, other.entries().get(), _entries.get(), dim(), result.entries().get());
};

template <class T>
PlusAx<T, gpu::CudaDeleter<T[]>> DenseVector<T, gpu::CudaDeleter<T[]>>::plusAx(const T& scalar, const DenseVector<T, gpu::CudaDeleter<T[]>>& other) const {
    return PlusAx<T, gpu::CudaDeleter<T[]>>(*this, scalar, other);
}

#endif

#endif
