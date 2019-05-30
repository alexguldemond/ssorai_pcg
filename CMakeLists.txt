cmake_minimum_required(VERSION 3.1)

project(PCG LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Werror -Wall -Wextra -pedantic")
list(APPEND CUDA_NVCC_FLAGS "-Werror -Wall -Wextra -pedantic -lineinfo")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(
		${PROJECT_SOURCE_DIR}/include
)
add_executable(pcg "${PROJECT_SOURCE_DIR}/src/Main.cpp")
add_executable(dotprod "${PROJECT_SOURCE_DIR}/src/Dotprod.cu")
add_executable(matvec "${PROJECT_SOURCE_DIR}/src/MatVec.cu")
add_executable(mattest "${PROJECT_SOURCE_DIR}/src/MatVecTest.cu")
add_executable(gpu "${PROJECT_SOURCE_DIR}/src/Parallel.cu")
target_link_libraries(gpu -lboost_date_time)
target_link_libraries(pcg -lboost_date_time)
