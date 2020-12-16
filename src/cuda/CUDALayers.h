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

#ifndef CUDALAYER_H_INCLUDE
#define CUDALAYER_H_INCLUDE
#ifdef USE_CUDA
#include "cuda/CUDACommon.h"
#include "Board.h"

#include <vector>
#include <array>

namespace CUDA_Backend {

static constexpr auto CONV_WIDTH = Board::WIDTH;
static constexpr auto CONV_HEIGHT = Board::HEIGHT;

class Convolve {
public:
    Convolve() = default;
    Convolve(const size_t batch, const size_t filter,
             const size_t in_channels, const size_t out_channels);
    ~Convolve();

    void Forward(const int batch, float *input, float *output,
                 void *scratch, size_t scratch_size, CudaHandel *handel);

    void LoadingWeight(const std::vector<float> &weights,
                       size_t &scratch_size);

    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases,
                       size_t &scratch_size);

private:
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;

    int m_filter_dim;
    int m_maxbatch;
    int m_filter;
    int m_in_channels;
    int m_out_channels;

#ifdef USE_CUDNN
    cudnnFilterDescriptor_t filter_desc;
    cudnnTensorDescriptor_t in_tensor_desc;
    cudnnTensorDescriptor_t out_tensor_desc;

    cudnnConvolutionDescriptor_t conv_desc;

    cudnnTensorDescriptor_t bias_desc;
    cudnnConvolutionFwdAlgo_t conv_algo;

    bool cudnn_applied{false};
#endif

    bool is_loaded{false};
    float *cuda_weights;
    float *cuda_biases{nullptr};
};

class FullyConnect {
public:
    FullyConnect() = default;
    FullyConnect(const size_t batch, const size_t inputs, 
                     const size_t outputs, bool ReLU);
    ~FullyConnect();

    void Forward(const int batch,
                 float *input,
                 float *output,
                 CudaHandel *handel);


    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases);
private:
    bool m_ReLU;
    int m_maxbatch;
    int m_inputs;
    int m_outputs;

    bool is_loaded{false};
    float *cuda_weights;
    float *cuda_biases;
};

class GlobalAvgPool {
public:
    GlobalAvgPool() = default; 
    GlobalAvgPool(const size_t batch,
                  const size_t channels);

    void Forward(const int batch, float *input, float *output);

private:
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;

    int m_maxbatch;
    int m_channels;
};

class SEUnit {
public:
    SEUnit() = default;
    SEUnit(const size_t batch, const size_t channels, const size_t se_size);
    ~SEUnit();

    void LoadingWeight(const std::vector<float> &weights_w1,
                       const std::vector<float> &weights_b1,
                       const std::vector<float> &weights_w2,
                       const std::vector<float> &weights_b2);

    void Forward(const int batch, float *input, float *output, CudaHandel *handel);
 
private:
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;

    int m_se_size;
    int m_maxbatch;
    int m_channels;

    bool is_loaded{false};
    std::array<float *, 3> cuda_op;

    float *cuda_weights_w1;
    float *cuda_weights_b1;
    float *cuda_weights_w2;
    float *cuda_weights_b2;
};

// class InputPool {
// public:
//     InputPool() = default;
//     InputPool(const size_t conv_size, const size_t batch,
//                   const size_t input_size, const size_t channels);
//     ~InputPool();
//
//     void LoadingWeight(const std::vector<float> &weights_w,
//                       const std::vector<float> &weights_b); 
//
//     void Forward(const int batch, float *input, float *output, CudaHandel *handel);
//
//     void set_convsize(const size_t conv_size);
//  
// private:
//     int width;
//     int height;
//     int spatial_size;
//
//     int m_maxbatch;
//     int m_input_size;
//     int m_channels;
//
//     bool is_loaded{false};
//
//     float *cuda_op;
//     float *cuda_weights_w;
//     float *cuda_weights_b;
//
// };
} // namespace CUDA_Backend

#endif
#endif
