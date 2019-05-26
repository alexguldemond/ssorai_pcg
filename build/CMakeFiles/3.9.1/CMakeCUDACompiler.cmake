set(CMAKE_CUDA_COMPILER "/usr/local/cuda-9.2/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "9.2.88")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "98")
set(CMAKE_CUDA_SIMULATE_ID "")
set(CMAKE_CUDA_SIMULATE_VERSION "")


set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda-9.2/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "cudadevrt;cudart_static;rt;pthread;dl")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs;/usr/local/cuda-9.2/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "cudadevrt;cudart_static;rt;pthread;dl;stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/usr/local/cuda-9.2/targets/x86_64-linux/lib/stubs;/usr/local/cuda-9.2/targets/x86_64-linux/lib;/opt/gnu/gcc/lib/gcc/x86_64-unknown-linux-gnu/4.9.2;/opt/gnu/gcc/lib64;/lib64;/usr/lib64;/opt/intel/composer_xe_2013_sp1.2.144/compiler/lib/intel64;/opt/intel/composer_xe_2013_sp1.2.144/ipp/lib/intel64;/opt/intel/composer_xe_2013_sp1.2.144/mkl/lib/intel64;/opt/intel/composer_xe_2013_sp1.2.144/tbb/lib/intel64/gcc4.4;/opt/gnu/gcc/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
