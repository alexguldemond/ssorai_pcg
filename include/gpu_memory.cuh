#ifndef GPU_MEMORY_CUH
#define GPU_MEMORY_CUH

#include "Error.cuh"
#include <memory>
#include <utility>
#include <algorithm>

namespace gpu {

    class stream {
    public:
	cudaStream_t _stream = nullptr;
	
	stream() {
	    cudaStreamCreate(&_stream);
	}

	stream(unsigned int flags) {
	    cudaStreamCreateWithFlags(&_stream, flags);
	}
	
	stream(const stream& other) = delete;
	
	stream(stream&& other) {
	    _stream = other._stream;
	    other._stream = nullptr;
	}
	
	~stream() {
	    if (_stream != nullptr) {
		cudaStreamDestroy(_stream);
	    }
	}
	
	stream& operator=(const stream& other) = delete;
	
	stream& operator=(stream&& other) {
	    if (&other != this) {
		std::swap(_stream, other._stream);
	    }
	    return *this;
	}
    };
    
    
    template<class T>
    T* safe_malloc(int count = 1) {
	T* ptr;
	checkCuda(cudaMalloc(&ptr, count * sizeof(T)));
	return ptr;
    }
    
    template<class T>
    T* safe_host_malloc(int count = 1) {
	T* ptr;
	checkCuda(cudaMallocHost(&ptr, count * sizeof(T)));
	return ptr;
    }
    
    template<class T>
    void safe_free(T* ptr) {
	if (ptr != nullptr) {
	    checkCuda(cudaFree(ptr));
	}
    }
    
    template<class T>
    void safe_host_free(T* ptr) {
	if (ptr != nullptr) {
	    checkCuda(cudaFreeHost(ptr));
	}
    }
    
    template<class T>
    class CudaDeleter {
    public:
	void operator()(T* ptr) {
	    safe_free(ptr);
	}
    };
    
    template<class T>
    class CudaHostDeleter {
    public:
	void operator()(T* ptr) {
	    safe_host_free(ptr);
	}
    };
    
    template<class T>
    class CudaDeleter<T[]> {
    public:
	void operator()(T* ptr) {
	    safe_free(ptr);
	}
    };
    
    template<class T>
    class CudaHostDeleter<T[]> {
    public:
	void operator()(T* ptr) {
	    safe_host_free(ptr);
	}
    };
    
    template<class T>
    void memcpy_to_host(T* host, const T* device, int count = 1) {
	checkCuda(cudaMemcpy(host, device, count*sizeof(T), cudaMemcpyDeviceToHost));
    }
    
    template<class T>
    void memcpy_to_device(const T* host, T* device, int count = 1) {
	checkCuda(cudaMemcpy(device, host, count*sizeof(T), cudaMemcpyHostToDevice));
    }
    
    template<class T>
    void memcpy_to_device_async(const T* host, T* device, cudaStream_t stream = 0, int count = 1) {
	checkCuda(cudaMemcpyAsync(device, host, count*sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    
    template<class T, class HostDeleter, class DeviceDeleter>
    void memcpy_to_host(std::unique_ptr<T, HostDeleter>& host, const std::unique_ptr<T, DeviceDeleter>& device) {
	memcpy_to_host(host.get(), device.get());
    }
    
    template<class T, class HostDeleter, class DeviceDeleter>
    void memcpy_to_host(std::unique_ptr<T[], HostDeleter>& host, const std::unique_ptr<T[], DeviceDeleter>& device, int count) {
	memcpy_to_host(host.get(), device.get(), count);
    }
    
    template<class T, class HostDeleter, class DeviceDeleter>
    void memcpy_to_device(const std::unique_ptr<T, HostDeleter>& host, std::unique_ptr<T, DeviceDeleter>& device) {
	memcpy_to_device(host.get(), device.get());
    }
    
    template<class T, class HostDeleter, class DeviceDeleter>
    void memcpy_to_device_async(const std::unique_ptr<T, HostDeleter>& host, std::unique_ptr<T, DeviceDeleter>& device, const stream& stream) {
	memcpy_to_device_async(host.get(), device.get(), stream._stream);
    }
    
    template<class T, class HostDeleter, class DeviceDeleter>
    void memcpy_to_device(const std::unique_ptr<T[], HostDeleter>& host, std::unique_ptr<T[], DeviceDeleter>& device, int count) {
	memcpy_to_device(host.get(), device.get(), count);
    }
    
    template<class T, class HostDeleter, class DeviceDeleter>
    void memcpy_to_device(const std::unique_ptr<T[], HostDeleter>& host, std::unique_ptr<T[], DeviceDeleter>& device, int count, const stream& stream) {
	memcpy_to_device_async(host.get(), device.get(), stream._stream, count);
    }
    
    template<class T>
    using device_ptr = std::unique_ptr<T, CudaDeleter<T>>;
    
    template<class T>
    using host_ptr = std::unique_ptr<T, CudaHostDeleter<T>>;
    
    template<class T>
    device_ptr<T> make_device() {
	T* ptr = safe_malloc<T>(1);
	return device_ptr<T>(std::move(ptr), CudaDeleter<T>());
    }
    
    template<class T>
    device_ptr<T[]> make_device(int count) {
	T* ptr = safe_malloc<T>(count);
	return device_ptr<T[]>(std::move(ptr), CudaDeleter<T[]>());
    }
    
    template<class T, class HostDeleter = std::default_delete<T>>
    device_ptr<T> make_device(const std::unique_ptr<T, HostDeleter>& host) {
        auto device = make_device<T>();
	memcpy_to_device(host, device);
	return device;
    }
    
    template<class T, class HostDeleter = std::default_delete<T[]>>
    device_ptr<T[]> make_device(const std::unique_ptr<T[], HostDeleter>& host, int count) {
        auto device = make_device<T>(count);
	memcpy_to_device(host, device, count);
	return device;
    }
    
    template<class T, class HostDeleter = std::default_delete<T>>
    device_ptr<T> make_device_async(const std::unique_ptr<T, HostDeleter>& host, const stream& stream) {
        auto device = make_device<T>();
	memcpy_to_device_async(host, device, stream);
	return device;
    }
    
    template<class T, class HostDeleter = std::default_delete<T[]>>
    device_ptr<T[]> make_device_async(const std::unique_ptr<T[], HostDeleter>& host, int count, const stream& stream) {
        auto device = make_device<T>(count);
	memcpy_to_device_async(host, device, count);
	return device;
    }
    
    template <class T>
    host_ptr<T> make_host() {
	T* ptr = safe_host_malloc<T>(1);
	return host_ptr<T>(std::move(ptr), CudaHostDeleter<T>());
    }
    
    template <class T>
    host_ptr<T[]> make_host(int count) {
	T* ptr = safe_host_malloc<T>(count);
	return host_ptr<T>(std::move(ptr), CudaHostDeleter<T[]>());
    }
    
    template <class T>
    T get_from_device(const device_ptr<T>& device) {
	T host;
	memcpy_to_host(&host, device.get());
	return host;
    }

    template <class T, class HostDeleter = std::default_delete<T[]>>
    std::unique_ptr<T[], HostDeleter> get_from_device(const device_ptr<T[]>& device, int count) {
	T* host = new T[count];
	memcpy_to_host(host, device.get(), count);
	return std::unique_ptr<T[], HostDeleter>(host, HostDeleter());
    }
    
}

#endif
