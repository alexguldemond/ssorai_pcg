#ifndef DEVICE_PTR_CUH
#define DEVICE_PTR_CUH

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

    DevicePtr(DevicePtr<T>&& other)=delete;
    
    DevicePtr<T>& operator=(const DevicePtr<T>& other)=delete;

    DevicePtr<T>& operator=(DevicePtr<T>&& other)=delete;

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

    int size() const {
	return count;
    }
    
    ~DevicePtr() {
	cudaFree(_devicePtr);
    }
    
};

#endif