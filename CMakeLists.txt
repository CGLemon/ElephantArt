cmake_minimum_required(VERSION 3.9)
project(ElephantArt)

if(DEBUG_MODE)
    message(STATUS "Debug mode")
    set(CMAKE_BUILD_TYPE DEBUG)
else()
    set(CMAKE_BUILD_TYPE RELEASE)
endif()

if (USE_AVX) 
    set(CMAKE_CXX_FLAGS "-mavx -mfma ${CMAKE_CXX_FLAGS}")
endif()

if(USE_FAST_PARSER)
    message(STATUS "Using fast parser.")
    add_definitions(-DUSE_FAST_PARSER)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/fast_float/include/fast_float)
endif()

if(GPU_BACKEND STREQUAL "CUDA")
    message(STATUS "Using CUDA backend.")
    add_definitions(-DUSE_CUDA)
    enable_language(CUDA)
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/cuda CUDA_SRCS)
    set(CMAKE_CUDA_FLAGS "" CACHE STRING "")
    if(CMAKE_CUDA_FLAGS)
        list(REMOVE_ITEM CMAKE_CUDA_FLAGS "-cudart static")
    endif()
    string(APPEND CMAKE_CUDA_FLAGS "-cudart shared")

    set(CUDA_STANDARD 14)
    set(CMAKE_CUDA_FLAGS
        "-gencode arch=compute_53,code=sm_53 -gencode arch=compute_70,code=sm_70 -gencode  arch=compute_75,code=compute_75"
    )
    set(CMAKE_CUDA_FLAGS "-Xptxas -O3 -Xcompiler -fPIC ${CMAKE_CUDA_FLAGS}")
else()
    message(STATUS "Using CPU backend.")
    if (USE_CUDNN)
        message(WARNING "Using cuDNN without CUDA. Please add flag -DGPU_BACKEND=CUDA.")
    endif()
endif()

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)

set(IncludePath "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(GCC_COMPILE_FLAGS "-Wall -Wextra -ffast-math -march=native -flto")
set(GCC_DISABLED_WARNING_COMPILE_FLAGS "-Wno-ignored-attributes -Wno-maybe-uninitialized -Wno-mismatched-tags")
set(CMAKE_CXX_FLAGS "${GCC_COMPILE_FLAGS} ${GCC_DISABLED_WARNING_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "-g -O3 -DNDEBUG ${CMAKE_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "-flto -g")

if (BLAS_BACKEND STREQUAL "EIGEN")
    if(NOT GPU_BACKEND STREQUAL "CUDA")
        message(STATUS "The CPU backend based on eigen")
    endif()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/Eigen)
    add_definitions(-DUSE_EIGEN)
    add_definitions(-DUSE_BLAS)
elseif(BLAS_BACKEND STREQUAL "OPENBLAS") 
    if(NOT GPU_BACKEND STREQUAL "CUDA")
        message(STATUS "The CPU backend based on openblas")
    endif()
    message(" Looking for system BLAS/OpenBLAS library.")
    find_package(BLAS REQUIRED)
    find_path(
        BLAS_INCLUDE_DIRS openblas_config.h
        /usr/include
        /usr/local/include
        /usr/include/openblas
        /opt/OpenBLAS/include
        /usr/include/x86_64-linux-gnu
        $ENV{BLAS_HOME}/include)
    add_definitions(-DUSE_OPENBLAS)
    add_definitions(-DUSE_BLAS)

    if(APPLE OR BLAS_INCLUDE_DIRS)
        message(" BLAS/OpenBLAS library was found.")
    else()
        message(ERROR "${ColorBoldRed} OpenBlas was NOT found, specify BLAS_INCLUDE_DIRS to indicate where it is.")
    endif()

    if((UNIX AND NOT APPLE) OR WIN32)
        include_directories(${BLAS_INCLUDE_DIRS})
    endif()
    if(APPLE)
        include_directories("/System/Library/Frameworks/Accelerate.framework/Versions/Current/Headers")
    endif()
else()
    if(NOT GPU_BACKEND STREQUAL "CUDA")
        message(STATUS "The CPU backend based on built-in blas")
    endif()
endif()

include_directories(${IncludePath})
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRCS)
add_executable(Elephant ${DIR_SRCS} ${CUDA_SRCS})

target_link_libraries(Elephant Threads::Threads)
target_link_libraries(Elephant ${BLAS_LIBRARIES})

if(GPU_BACKEND STREQUAL "CUDA")
    target_compile_definitions(Elephant PRIVATE USE_CUDA_BACKEND)
    find_package(CUDA REQUIRED)
    if (USE_CUDNN)
        message(STATUS "The CUDA backend based on cuDNN")
        message(" Looking for cuDNN library...")
        find_path(CUDNN_INCLUDE_DIR cudnn.h HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES cuda/include include)
        if((NOT CUDNN_INCLUDE_DIR))
            message(ERROR "${ColorBoldRed} cudnn.h was NOT found, specify CUDNN_INCLUDE_DIR to indicate where it is.")
        else()
            message(" CuDNN library was found.")
        endif()
        add_definitions(-DUSE_CUDNN)
        find_library(CUDNN_LIBRARY libcudnn.so PATHS /usr/local/cuda/lib64 /opt/cuda/lib64)
        include_directories(SYSTEM ${CUDNN_INCLUDE_DIR})
    endif()

    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
    target_link_libraries(Elephant ${CUDNN_LIBRARY} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})
endif()
