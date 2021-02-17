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

#ifndef BLAS_H_INCLUDE
#define BLAS_H_INCLUDE

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef USE_OPENBLAS
#ifndef __APPLE__
#include <cblas.h>
#endif
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

#include "Board.h"

static constexpr auto CONV_WIDTH = Board::WIDTH;
static constexpr auto CONV_HEIGHT = Board::HEIGHT;

template <bool TA, bool TB> 
class Gemm {
public:
    static void apply(int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc);
};

// TODO: All merge in one blas
class Blas {
public:
    // For convolution
    static void fixed_gemm(const int M, const int N, const int K,
                           const float alpha, 
                           const float *A, const int lda,
                           const float *B, const int ldb,
                           const float beta,
                           float *C, const int ldc);
    // For Winograd
    static void winograd_gemm(const int set_U, const int set_V, const int set_M,
                              const int M, const int N, const int K,
                              const float alpha,
                              const float *A, const int lda,
                              const float *B, const int ldb,
                              const float beta,
                              float *C, const int ldc);

    // For fullyconnect
    static void dense(const int inputs,
                      const int outputs,
                      const int batch_size,
                      const float *input,
                      const float *kernel,
                      float *output);

};

class FullyConnect {
public:
    FullyConnect() = delete;
    static void Forward(const int inputs_size,
                        const int outputs_size,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        const std::vector<float> &biases,
                        std::vector<float> &output,
                        const bool ReLU);

    static std::vector<float> innerproduct(const int inputs_size,
                                           const int outputs_size,
                                           const std::vector<float> &input,
                                           const std::vector<float> &weights,
                                           const std::vector<float> &biases,
                                           const bool ReLU);
};

class Convolve1 {
public:
    Convolve1() = delete;
    static void Forward(const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &output);

private:
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;
};

template <size_t FILTER_SIZE>
class Convolve {
public:
    Convolve() = delete;
    static void Forward(const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &col,
                        std::vector<float> &output);

    static size_t get_workspace_size(const size_t input_channels);

private:
    static void im2col(const int channels,
                       const std::vector<float> &input,
                       std::vector<float> &col);

    static constexpr auto filter_size = FILTER_SIZE;
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;
};


class AddSpatialBias {
public:
    AddSpatialBias() = delete;
    static void Forward(const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &biases);
private:
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;
};

class Batchnorm {
public:
    Batchnorm() = delete;
    static void Forward(const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &means,
                        const std::vector<float> &stddevs,
                        const float *const eltwise = nullptr,
                        const bool ReLU = true);

private:
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;
};

class GlobalAvgPool {
public:
    GlobalAvgPool() = delete;
    static void Forward(const size_t input_channels,
                        const std::vector<float> &input,
                        std::vector<float> &output);

private:
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;

};

class SEUnit {
public:
    SEUnit() = delete;
    static void Forward(const size_t channels,
                        const size_t se_size,
                        std::vector<float> &input,
                        const std::vector<float> &residual,
                        const std::vector<float> &weights_w1,
                        const std::vector<float> &weights_b1,
                        const std::vector<float> &weights_w2,
                        const std::vector<float> &weights_b2);

private:
    static void SEProcess(const size_t channels,
                          std::vector<float> &input,
                          const std::vector<float> &residual,
                          const std::vector<float> &scale);

    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;
};

class Activation {
public:
    static std::vector<float> Softmax(const std::vector<float> &input,
                                      const float temperature = 1.0f);

    static std::vector<float> Tanh(const std::vector<float> &input);

    static std::vector<float> Sigmoid(const std::vector<float> &input);
};




template <size_t FILTER_SIZE>
void Convolve<FILTER_SIZE>::Forward(const size_t input_channels,
                                    const size_t output_channels,
                                    const std::vector<float> &input,
                                    const std::vector<float> &weights,
                                    std::vector<float> &col,
                                    std::vector<float> &output) {

    constexpr auto filter_len = filter_size * filter_size;
    const auto filter_dim = filter_len * input_channels;
    assert(output_channels * spatial_size == output.size());

    im2col(input_channels, input, col);
    Blas::fixed_gemm((int)output_channels,
                     spatial_size,
                     (int)filter_dim,
                     1.0f,
                     weights.data(),
                     (int)filter_dim,
                     col.data(),
                     spatial_size,
                     0.0f,
                     output.data(),
                     spatial_size);

}

template <size_t FILTER_SIZE>
void Convolve<FILTER_SIZE>::im2col(const int channels,
                                   const std::vector<float> &input,
                                   std::vector<float> &output) {

    constexpr int pad = (filter_size / 2);
    unsigned int output_h = height + 2 * pad - filter_size + 1;
    unsigned int output_w = width + 2 * pad - filter_size + 1;

    const float *data_im = input.data();
    float *data_col = output.data();

    for (int channel = channels; channel--; data_im += spatial_size) {
        for (unsigned int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
            for (unsigned int kernel_col = 0; kernel_col < filter_size;  kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (unsigned(input_row) < height) {
                        int input_col = -pad + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (unsigned(input_col) < width) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col++;
                        }
                    } else {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    }
                    input_row++;
                }
            }
        }
    }
}

template <size_t FILTER_SIZE>
size_t Convolve<FILTER_SIZE>::get_workspace_size(const size_t input_channels) {
    constexpr  auto filter_len = filter_size * filter_size;
    const auto filter_dim = filter_len * input_channels;
    return filter_dim * width * height;
}

class InputPool {
public:
    InputPool() = delete;
    static void Forward(const size_t input_size,
                        const size_t squeeze,
                        const size_t channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights_w1,
                        const std::vector<float> &weights_b1,
                        const std::vector<float> &weights_w2,
                        const std::vector<float> &weights_b2,
                        std::vector<float> &output);

private:
    static constexpr auto width = CONV_WIDTH;
    static constexpr auto height = CONV_HEIGHT;
    static constexpr auto spatial_size = width * height;
};


#endif
