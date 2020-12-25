/*
    This file is part of Saya.
    Copyright (C) 2020 Hung-Zhe Lin

    Saya is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Saya is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Saya.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef USE_CUDA

#include "cuda/CUDACommon.h"
#include "Utils.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace CUDA {

const char* cublasGetErrorString(cublasStatus_t status) {
    switch(status) {
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}

void CublasError(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        const char *cause = cublasGetErrorString(status);
        auto err = std::ostringstream{};
        err << "CUBLAS error: " << cause;
        throw std::runtime_error(err.str());
    }
}

void CudaError(cudaError_t status) {
  if (status != cudaSuccess) {
        const char *cause = cudaGetErrorString(status);
        auto err = std::ostringstream{};
        err << "CUDA Error: " << cause;
        throw std::runtime_error(err.str());
  }
}

int get_devicecount() {
    int n = 0;
    cudaError_t status = cudaGetDeviceCount(&n);
    ReportCUDAErrors(status);
    return n;
}

int get_device(int n) {
    cudaError_t status = cudaGetDevice(&n);
    ReportCUDAErrors(status);
    return n;
}

#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        const char *s = cudnnGetErrorString(status);
        std::cerr << "CUDA Error: " << s << "\n";
        exit(-1);
    }
}

cudnnHandle_t cudnn_handle(int n) {
    static int init[MAX_SUPPORT_GPUS] = {0};
    static cudnnHandle_t handle[MAX_SUPPORT_GPUS];
    int i = get_device(n);
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

cublasHandle_t blas_handle(int n) {
    static int init[MAX_SUPPORT_GPUS] = {0};
    static cublasHandle_t handle[MAX_SUPPORT_GPUS];
    int i = get_device(n);
    if (!init[i]) {
        cublasCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}

void CudaHandel::apply(int n) {
#ifdef USE_CUDNN
  cudnn_handel = cudnn_handle(n);
#endif
  cublas_handel = blas_handle(n);
}

bool is_using_cuDNN() {
#ifdef USE_CUDNN
    return true;
#else
    return false;
#endif
}

void output_spec(const cudaDeviceProp & sDevProp) {
    Utils::auto_printf(" Device name: %s\n", sDevProp.name);
    Utils::auto_printf(" Device memory(MiB): %zu\n", (sDevProp.totalGlobalMem/(1024*1024)));
    Utils::auto_printf(" Memory per-block(KiB): %zu\n", (sDevProp.sharedMemPerBlock/1024));
    Utils::auto_printf(" Register per-block(KiB): %zu\n", (sDevProp.regsPerBlock/1024));
    Utils::auto_printf(" Warp size: %zu\n", sDevProp.warpSize);
    Utils::auto_printf(" Memory pitch(MiB): %zu\n", (sDevProp.memPitch/(1024*1024)));
    Utils::auto_printf(" Constant Memory(KiB): %zu\n", (sDevProp.totalConstMem/1024));
    Utils::auto_printf(" Max thread per-block: %zu\n", sDevProp.maxThreadsPerBlock);
    Utils::auto_printf(" Max thread dim: (%zu, %zu, %zu)\n", sDevProp.maxThreadsDim[0], sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2]);
    Utils::auto_printf(" Max grid size: (%zu, %zu, %zu)\n", sDevProp.maxGridSize[0], sDevProp.maxGridSize[1], sDevProp.maxGridSize[2]);
    Utils::auto_printf(" Clock: %zu(kHz)\n", (sDevProp.clockRate/1000));
    Utils::auto_printf(" textureAlignment: %zu\n", sDevProp.textureAlignment);
}

void gpu_info() {
    int devicecount = get_devicecount();
    Utils::auto_printf("Number of CUDA devices: %zu\n", devicecount);

    if(devicecount == 0) {
        throw std::runtime_error("No CUDA device");
    }

    for(int i = 0; i < devicecount; ++i) {
        Utils::auto_printf("\n=== Device %zu ===\n", i);
        cudaDeviceProp sDeviceProp;
        cudaGetDeviceProperties(&sDeviceProp, i);
        output_spec(sDeviceProp);
    }
    Utils::auto_printf("\n");
    // cudaSetDevice(0);
}
} // namespace CUDA

#endif
