#ifndef DEVICE_PTR_CUH
#define DEVICE_PTR_CUH

#include <vector>
#include <algorithm>
#include "Error.cuh"

template <class T>
class DevicePtr {
private:
    int count;
    T* _devicePtr;
    
public:
    DevicePtr() : count(1) {
	checkCuda(cudaMalloc(&_devicePtr, count*sizeof(float)));
    }

    DevicePtr(int n) : count(n) {
	checkCuda(cudaMalloc(&_devicePtr, count*sizeof(float)));
    }

    DevicePtr(const T* data): count(1) {
	checkCuda(cudaMalloc(&_devicePtr, count*sizeof(float)));
	assignFromHost(data);
    }
    
    DevicePtr(const T* data, int dataCount): count(dataCount) {
	checkCuda(cudaMalloc(&_devicePtr, count*sizeof(float)));
	assignFromHost(data);
    }

    DevicePtr(const DevicePtr<T>& other)=delete;

    DevicePtr(DevicePtr<T>&& other) {
	//Deliberately done via swap to encourage reuse of memory
	int temp = count;
	count = other.count;
	other.count = temp;
	std::swap(other._devicePtr, _devicePtr);
    }
    
    DevicePtr<T>& operator=(const DevicePtr<T>& other)=delete;

    DevicePtr<T>& operator=(DevicePtr<T>&& other) {
	if (this != &other) {
	    int temp = count;
	    count = other.count;
	    other.count = temp;
	    std::swap(other._devicePtr, _devicePtr);
	}
	return *this;
    }

    T* operator->() {
	return _devicePtr;
    }

    T* raw() {
	return _devicePtr;
    }

    const T* raw() const {
	return _devicePtr;
    }

    void copyToHost(T* hostPtr) const {
	checkCuda(cudaMemcpy(hostPtr, _devicePtr, count*sizeof(T), cudaMemcpyDeviceToHost));
    }

    DevicePtr<T>& assignFromHost(const T* hostPtr) {
	checkCuda(cudaMemcpy(_devicePtr, hostPtr, count*sizeof(T), cudaMemcpyHostToDevice));
	return (*this);
    }

    T get() const {
	T t;
	copyToHost(&t);
	return t;
    }

    std::vector<T> getAll() const {
	std::vector<T> t(count);
	copyToHost(&t[0]);
	return t;
    }
    
    int size() const {
	return count;
    }
    
    ~DevicePtr() {
	if (_devicePtr != nullptr) {
	    cudaFree(_devicePtr);
	}
    }
    
};

#endif