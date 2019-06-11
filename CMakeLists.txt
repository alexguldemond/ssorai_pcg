cmake_minimum_required(VERSION 3.1)

project(PCG LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Werror -Wall -Wextra -pedantic -O3")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Werror -Wall -Wextra -pedantic -O3")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(
		${PROJECT_SOURCE_DIR}/include
		)

add_executable(vec "${PROJECT_SOURCE_DIR}/src/VecTest.cpp")
add_executable(vec_gpu "${PROJECT_SOURCE_DIR}/src/VecTestGpu.cu")	      
add_executable(mat "${PROJECT_SOURCE_DIR}/src/MatTest.cpp")
add_executable(pcg "${PROJECT_SOURCE_DIR}/src/Main.cu")
add_executable(ssora "${PROJECT_SOURCE_DIR}/src/SsoraInveser.cpp")
add_executable(dotprod "${PROJECT_SOURCE_DIR}/src/Dotprod.cu")
add_executable(matvec "${PROJECT_SOURCE_DIR}/src/MatVec.cu")
add_executable(ssora_gpu "${PROJECT_SOURCE_DIR}/src/SsoraiGpu.cu")
add_executable(build_test "${PROJECT_SOURCE_DIR}/src/BuildMatrix.cu")
target_link_libraries(pcg -lboost_date_time -lboost_program_options)
