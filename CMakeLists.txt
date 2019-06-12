cmake_minimum_required(VERSION 3.1)

project(PCG LANGUAGES CUDA CXX)

## Download GTest
# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
  include_directories("${gtest_SOURCE_DIR}/include")
endif()

## My Stuff
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

add_executable(pcg "${PROJECT_SOURCE_DIR}/src/Main.cu")
target_link_libraries(pcg -lboost_date_time -lboost_program_options)

## Tests
enable_testing()
add_executable(buildMatrix "${PROJECT_SOURCE_DIR}/test/BuildMatrix.cu")
add_executable(dotprod "${PROJECT_SOURCE_DIR}/test/Dotprod.cu")
add_executable(matvec "${PROJECT_SOURCE_DIR}/test/MatVec.cu")
add_executable(ssora_gpu "${PROJECT_SOURCE_DIR}/test/SsoraiGpu.cu")
add_executable(vec_gpu "${PROJECT_SOURCE_DIR}/test/VecTestGpu.cu")	      

target_link_libraries(buildMatrix gtest_main)
target_link_libraries(dotprod gtest_main)
target_link_libraries(matvec gtest_main)
target_link_libraries(ssora_gpu gtest_main)
target_link_libraries(vec_gpu gtest_main)

add_test(NAME buildMatrix COMMAND buildMatrix)
add_test(NAME dotprod COMMAND dotprod)
add_test(NAME matvec COMMAND matvec)
add_test(NAME ssora_gpu COMMAND ssora_gpu)
add_test(NAME vec_gpu COMMAND vec_gpu)
