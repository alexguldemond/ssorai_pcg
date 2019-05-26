#ifndef DENSEVECTOR_HPP
#define DENSEVECTOR_HPP

#include <vector>
#include <string>
#include <sstream>

template <class T>
class DenseVector {
private:
    std::vector<T>  entries;

public:

    explicit DenseVector(const std::vector<T>& entries): entries(entries) {
	//Intentionally empty
    }

    explicit DenseVector(std::vector<T>&& entries):  entries(entries) {
	//Intentionally empty
    }
    
    static DenseVector<T> zero(int dim);

    static DenseVector<T> constant(int dim, const T& value);
    
    int dim() const;

    T* data() {
	return entries.data();
    }

    T dot(const DenseVector<T>& other) const;

    T normSquared() const;

    std::string toString() const;

    T operator()(int i) const;

    T operator[](int i) const;
    
    T& operator[](int i);

    DenseVector<T>& operator+=(const DenseVector<T>& other);

    DenseVector<T>& updateAx(const T& scalar, const DenseVector<T>& other);

    DenseVector<T> plusAx(const T& scalar, const DenseVector<T>& other) const;
};

template <class T>
DenseVector<T> DenseVector<T>::zero(int dim) {
    std::vector<T> vec(dim, 0);
    return DenseVector<T>(vec);
}

template <class T>
DenseVector<T> DenseVector<T>::constant(int dim, const T& value) {
    std::vector<T> vec(dim, value);
    return DenseVector<T>(vec);
}

template <class T>
inline int DenseVector<T>::dim() const  {
    return entries.size();
}

template <class T>
T DenseVector<T>::dot(const DenseVector<T>& other) const {
    double sum = 0.0;
    for (int i = 0; i < dim(); i++) {
	sum += this->entries[i] * other(i);
    }
    return sum;
}

template <class T>
T DenseVector<T>::normSquared() const {
    return this->dot(*this);
}

template <class T>
std::string DenseVector<T>::toString() const{
    std::stringstream stream;
    stream << "[ ";
    for (int i = 0; i < dim(); i++) {
	stream << entries[i] << " ";
    }
    stream << "]";
    return stream.str();
}

template <class T>
T DenseVector<T>::operator()(int i) const {
    return entries[i];
}

template <class T>
T DenseVector<T>::operator[](int i) const {
    return entries[i];
}

template <class T>
T& DenseVector<T>::operator[](int i) {
    return entries[i];
}

template <class T>
DenseVector<T>& DenseVector<T>::operator+=(const DenseVector<T>& other) {
    for (int i = 0; i < dim(); i++) {
	this->entries[i] += other(i);
    }
    return *this;
}

template <class T>
DenseVector<T>& DenseVector<T>::updateAx(const T& scalar, const DenseVector<T>& other) {
    for (int i = 0; i < dim(); i++) {
	this->entries[i] += scalar * other(i);
    }
    return *this;
}

template <class T>
DenseVector<T> DenseVector<T>::plusAx(const T& scalar, const DenseVector<T>& other) const {
    DenseVector<T> newVec(*this);
    for (int i = 0; i < dim(); i++) {
	newVec.entries[i] += scalar * other(i);
    }
    return newVec;
}

#endif
