#ifndef DENSEVECTOR_HPP
#define DENSEVECTOR_HPP

#include <memory>
#include <utility>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <functional>

template <class T,
	  class Deleter = std::default_delete<T[]> >
class DenseVector {
private:
    int _dim;

public:

    std::unique_ptr<T[], Deleter> entries;
    
    explicit DenseVector(int dim, const std::unique_ptr<T[], Deleter>& newEntries): _dim(dim) {
	T::operator new[](dim);
	T* vals = new T[dim];
	std::copy(newEntries.get(), newEntries.get() + _dim, vals);
	entries = std::unique_ptr<T[], Deleter>(vals, Deleter());
    }

    explicit DenseVector(int dim, std::unique_ptr<T[], Deleter>&& entries): _dim(dim),  entries(std::move(entries)) {
	//Intentionally empty
    }

    DenseVector(const DenseVector<T, Deleter>& other) : _dim(other._dim) {
	T* newEntries = new T[_dim];
	std::copy(other.entries.get(), other.entries.get() + _dim, newEntries);
	entries = std::unique_ptr<T[], Deleter>(newEntries, Deleter());
    }

    DenseVector(DenseVector<T, Deleter>&& other): _dim(other._dim), entries(std::move(other.entries)) {
	//Intentionally Empty
    }
    
    DenseVector<T, Deleter>& operator=(const DenseVector<T, Deleter>& other) {
	if (this != &other) {
	    _dim = other._dim;
	    T* newEntries = new T[_dim];
	    std::copy(newEntries, newEntries + _dim , other.entries);
	    entries.reset(newEntries);
	}
	return *this;
    }

    DenseVector<T, Deleter>& operator=(DenseVector<T, Deleter>&& other) {
	if (this != &other) {
	    _dim = other._dim;
	    entries.swap(other.entries);
	}
	return *this;
    }
    
    int dim() const;

    T* data() {
	return entries.get();
    }

    T dot(const DenseVector<T, Deleter>& other) const;

    T normSquared() const;

    std::string toString() const;

    T operator()(int i) const;

    const T& operator[](int i) const;
    
    T& operator[](int i);

    DenseVector<T, Deleter>& operator+=(const DenseVector<T, Deleter>& other);

    DenseVector<T, Deleter>& updateAx(const T& scalar, const DenseVector<T, Deleter>& other);

    DenseVector<T, Deleter> plusAx(const T& scalar, const DenseVector<T, Deleter>& other) const;
};

namespace DenseVectorFactory {		  

    template<class T>
    static T* allocate(int count) {
	return new T[count];
    }

    template <class T, class Deleter>
    DenseVector<T, Deleter> incremental(int dim, const T& startValue, Deleter deleter, std::function<T*(int)> allocater = allocate<T> ) {
	T* entries = allocater(dim);
	std::iota(entries, entries+dim, startValue);
	return DenseVector<T, Deleter>(dim, std::unique_ptr<T[], Deleter>(entries, deleter));
    }
    
    template <class T, class Deleter>
    DenseVector<T, Deleter> constant(int dim, const T& value, Deleter deleter, std::function<T*(int)> allocater = allocate<T> ) {
	T* entries = allocater(dim);
	std::fill(entries, entries+dim, value);
	return DenseVector<T, Deleter>(dim, std::unique_ptr<T[], Deleter>(entries, deleter));
    }
    
    template <class T, class Deleter>
    DenseVector<T, Deleter> zero(int dim, Deleter deleter, std::function<T*(int)> allocater = allocate<T>) {
	return constant<T, Deleter>(dim, 0, deleter, allocater);
    }

    template <class T >
    DenseVector<T> incremental(int dim, const T& startValue) {
	T* entries = new T[dim];
	std::iota(entries, entries+dim, startValue);
	return DenseVector<T>(dim, std::unique_ptr<T[]>(entries));
    }
    
    template <class T >
    DenseVector<T> constant(int dim, const T& value) {
	T* entries = new T[dim];
	std::fill(entries, entries+dim, value);
	return DenseVector<T>(dim, std::unique_ptr<T[]>(entries));
    }

    template <class T>
    DenseVector<T> zero(int dim) {
	return constant<T>(dim, 0);
    }
}

template <class T, class Deleter>
inline int DenseVector<T, Deleter>::dim() const  {
    return _dim;
}

template <class T, class Deleter>
T DenseVector<T, Deleter>::dot(const DenseVector<T, Deleter>& other) const {
    double sum = 0.0;
    for (int i = 0; i < dim(); i++) {
	sum += this->entries[i] * other[i];
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
	stream << entries[i] << " ";
    }
    stream << "]";
    return stream.str();
}

template <class T, class Deleter>
T DenseVector<T, Deleter>::operator()(int i) const {
    return entries[i];
}

template <class T, class Deleter>
const T& DenseVector<T, Deleter>::operator[](int i) const {
    return entries[i];
}

template <class T, class Deleter>
T& DenseVector<T, Deleter>::operator[](int i) {
    return entries[i];
}

template <class T, class Deleter>
DenseVector<T, Deleter>& DenseVector<T, Deleter>::operator+=(const DenseVector<T, Deleter>& other) {
    for (int i = 0; i < dim(); i++) {
	this->entries[i] += other[i];
    }
    return *this;
}

template <class T, class Deleter>
DenseVector<T, Deleter>& DenseVector<T, Deleter>::updateAx(const T& scalar, const DenseVector<T, Deleter>& other) {
    for (int i = 0; i < dim(); i++) {
	this->entries[i] += scalar * other[i];
    }
    return *this;
}

template <class T, class Deleter>
DenseVector<T, Deleter> DenseVector<T, Deleter>::plusAx(const T& scalar, const DenseVector<T, Deleter>& other) const {
    DenseVector<T, Deleter> newVec(*this);
    for (int i = 0; i < dim(); i++) {
	newVec.entries[i] += scalar * other[i];
    }
    return newVec;
}

#endif
