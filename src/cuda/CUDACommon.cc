/*
    This file is part of ElephantArt.
    Copyright (C) 2021 Hung-Zhe Lin

    ElephantArt is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElephantArt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElephantArt.  If not, see <http://www.gnu.org/licenses/>.
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

std::string output_spec(const cudaDeviceProp &dev_prop) {
    auto out = std::ostringstream{};

    out << " Device name: "             << dev_prop.name                       << '\n';
    out << " Device memory(MiB): "      << dev_prop.totalGlobalMem/(1024*1024) << '\n';
    out << " Memory per-block(KiB): "   << dev_prop.sharedMemPerBlock/1024     << '\n';
    out << " Register per-block(KiB): " << dev_prop.regsPerBlock/1024          << '\n';
    out << " Warp size: "               << dev_prop.warpSize                   << '\n';
    out << " Memory pitch(MiB): "       << dev_prop.memPitch/(1024*1024)       << '\n';
    out << " Constant Memory(KiB): "    << dev_prop.totalConstMem/1024         << '\n';
    out << " Max thread per-block: "    << dev_prop.maxThreadsPerBlock         << '\n';
    out << " Max thread dim: ("
            << dev_prop.maxThreadsDim[0] << ", "
            << dev_prop.maxThreadsDim[1] << ", "
            << dev_prop.maxThreadsDim[2] << ")\n";
    out << " Max grid size: ("
            << dev_prop.maxGridSize[0] << ", "
            << dev_prop.maxGridSize[1] << ", "
            << dev_prop.maxGridSize[2] << ")\n";
    out << " Clock: "             << dev_prop.clockRate/1000   << "(kHz)" << '\n';
    out << " Texture Alignment: " << dev_prop.textureAlignment << '\n';

    return out.str();
}

std::string check_devices() {
    auto out = std::ostringstream{};

    int devicecount = get_devicecount();
    if (devicecount == 0) {
        throw std::runtime_error("No CUDA device");
    }

    int cuda_version;
    cudaDriverGetVersion(&cuda_version);
    {
        const auto major = cuda_version/1000;
        const auto minor = (cuda_version - major * 1000)/10;
        out << "CUDA version: Major " << major << ", Minor " << minor << '\n';
    }

    out << "Using cuDNN: ";
    if (is_using_cuDNN()) {
        out << "Yes\n";
#ifdef USE_CUDNN
        const auto cudnn_version = cudnnGetVersion();
        const auto major = cudnn_version/1000;
        const auto minor = (cudnn_version -  major * 1000)/100;
        out << "cuDNN version: Major " << major << ", Minor " << minor << '\n';
#endif
    } else {
        out << "No\n";
    }

    out << "Number of CUDA devices: " << devicecount << '\n';
    for(int i = 0; i < devicecount; ++i) {
        out << "=== Device " << i << " ===\n";
        cudaDeviceProp sDeviceProp;
        cudaGetDeviceProperties(&sDeviceProp, i);
        out << output_spec(sDeviceProp);
    }
    out << std::endl;

    return out.str();
}
} // namespace CUDA

#endif
