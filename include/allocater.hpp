#ifndef ALLOCATER_HPP
#define ALLOCATER_HPP

#include <iostream>

template<class T, class Deleter>
struct Allocater {
    static T* allocate( int count) {
	return new T[count];
    }
};

#ifdef __CUDACC__
#include "gpu_memory.cuh"

template<class T>
struct Allocater<T, gpu::CudaDeleter<T[]>> {
    static T* allocate( int count) {
	return gpu::safe_malloc<T>(count);
    }
};

template <class T>
struct Allocater<T, gpu::CudaHostDeleter<T[]>> {
    static T* allocate( int count) {
	return gpu::safe_host_malloc<T>(count);
    }
};

template<class T>
struct Allocater<T, gpu::CudaDeleter<T>> {
    static T* allocate( int count) {
	return gpu::safe_malloc<T>();
    }
};

template <class T>
struct Allocater<T, gpu::CudaHostDeleter<T>> {
    static T* allocate( int count) {
	return gpu::safe_host_malloc<T>();
    }
};


#endif

#endif
