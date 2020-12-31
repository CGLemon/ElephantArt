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

#include "Blas.h"
#include <cmath>

#ifdef USE_EIGEN
// Eigen helpers
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
#endif

void gemm_nn(int M, int N, int K, float alpha, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float A_PART = alpha * A[i * lda + k];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float alpha, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
          float sum = 0;
          for (int k = 0; k < K; ++k) {
              sum += alpha * A[i * lda + k] * B[j * ldb + k];
          }
          C[i * ldc + j] += sum;
       }
    }
}

void gemm_tn(int M, int N, int K, float alpha, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            float A_PART = alpha * A[k * lda + i];
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] += A_PART * B[k * ldb + j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float alpha, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += alpha * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

#define INITIALIZE_GEMM(M, N, beta)   \
    for (int i = 0; i < M; ++i) {     \
        for (int j = 0; j < N; ++j) { \
            C[i * ldc + j] *= beta;   \
        }                             \
    }



template <>
void Gemm<false, false>::apply(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc) {
    INITIALIZE_GEMM(M, N, beta);
    gemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<true, false>::apply(int M, int N, int K,
                              float alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              float beta,
                              float *C, int ldc) {
    INITIALIZE_GEMM(M, N, beta);
    gemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<false, true>::apply(int M, int N, int K,
                              float alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              float beta,
                              float *C, int ldc) {
    INITIALIZE_GEMM(M, N, beta);
    gemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<true, true>::apply(int M, int N, int K,
                             float alpha,
                             const float *A, int lda,
                             const float *B, int ldb,
                             float beta,
                             float *C, int ldc) {
    INITIALIZE_GEMM(M, N, beta);
    gemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

#undef INITIALIZE_GEMM

void Blas::fixed_gemm(const int M, const int N, const int K,
                      const float alpha, 
                      const float *A, const int lda,
                      const float *B, const int ldb,
                      const float beta,
                      float *C, const int ldc) {

#ifndef USE_BLAS
    Gemm<false, false>::apply(M, N, K,
                              alpha,
                              A, lda,
                              B, ldb,
                              beta,
                              C, ldc);
#else
#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                alpha,
                A, lda,
                B, ldb,
                beta,
                C, ldc);

#endif
#ifdef USE_EIGEN
    (void) alpha;
    (void) beta;
    (void) lda;
    (void) ldb;
    (void) ldc;
    auto C_mat = EigenMatrixMap<float>(C, N, M);
    C_mat.noalias() = 
        ConstEigenMatrixMap<float>(B, N, K) *
        ConstEigenMatrixMap<float>(A, K, M);
#endif
#endif
}

void Blas::winograd_gemm(const int set_U, const int set_V, const int set_M,
                         const int M, const int N, const int K,
                         const float alpha,
                         const float *A, const int lda,
                         const float *B, const int ldb,
                         const float beta,
                         float *C, const int ldc) {

#ifndef USE_BLAS
    Gemm<true, false>::apply(M, N, K,
                             alpha,
                             A + set_U, lda,
                             B + set_V, ldb,
                             beta,
                             C + set_M, ldc);

#else
#ifdef USE_OPENBLAS
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K,
                alpha,
                A + set_U, lda,
                B + set_V, ldb,
                beta,
                C + set_M, ldc);

#endif
#ifdef USE_EIGEN
    (void) alpha;
    (void) beta;
    (void) lda;
    (void) ldb;
    (void) ldc;
    auto C_mat = EigenMatrixMap<float>(C + set_M, N, M);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(B + set_V, N, K) *
        ConstEigenMatrixMap<float>(A + set_U, M, K).transpose();

#endif
#endif
}

void Blas::dense(const int input_size,
                 const int output_size,
                 const int batch_size,
                 const float *inputs,
                 const float *kernel,
                 float *outputs) {

 static constexpr float alpha = 1.0f;
 static constexpr float beta = 0.0f;

#ifndef USE_BLAS
    Gemm<false, true>::apply(batch_size, output_size, input_size,
                             alpha,
                             inputs, input_size,
                             kernel, input_size,
                             beta,
                             outputs, output_size);
#else
#ifdef USE_OPENBLAS
    if (batch_size == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    output_size, input_size, 1.0f, kernel,
                    input_size, inputs, 1, 0.0f, outputs, 1);
    } else {
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    output_size, batch_size,  input_size,         
                    alpha,              
                    kernel, input_size,
                    inputs, input_size,
                    beta, 
                    outputs, output_size); 
  }

#endif
#ifdef USE_EIGEN
    (void) alpha;
    (void) beta;
    if (batch_size == 1) {
        EigenVectorMap<float> y(outputs, output_size);
        y.noalias() =
            ConstEigenMatrixMap<float>(kernel, input_size, output_size).transpose() *
            ConstEigenVectorMap<float>(inputs, input_size);
    } else {
        auto C_mat = EigenMatrixMap<float>(outputs, output_size, batch_size);
        C_mat.noalias() =
            ConstEigenMatrixMap<float>(kernel, input_size, output_size)
                .transpose() *
                ConstEigenMatrixMap<float>(inputs, input_size, batch_size);

  }
#endif
#endif
}

void FullyConnect::Forward(const int input_size,
                           const int output_size,
                           const std::vector<float> &input,
                           const std::vector<float> &weights,
                           const std::vector<float> &biases,
                           std::vector<float> &output,
                           const bool ReLU) {

    const auto lambda_ReLU = [](const auto val) -> float {
        return (val > 0.0f) ? val : 0.0f;
    };

    static constexpr int batch = 1;
    Blas::dense(input_size,
                output_size,
                batch,
                input.data(),
                weights.data(),
                output.data());

    if (ReLU) {
        for (auto o = int{0}; o < output_size; ++o) {
            output[o] = lambda_ReLU(biases[o] + output[o]);
        }
    } else {
        for (auto o = int{0}; o < output_size; ++o) {
            output[o] = biases[o] + output[o];
        }
    }
}

std::vector<float> FullyConnect::innerproduct(const int input_size,
                                              const int output_size,
                                              const std::vector<float> &input,
                                              const std::vector<float> &weights,
                                              const std::vector<float> &biases,
                                              const bool ReLU) {

    auto output = std::vector<float>(output_size);
    output.reserve(output_size);
    const auto lambda_ReLU = [](const auto val) -> float {
        return (val > 0.0f) ? val : 0.0f;
    };

    static constexpr int batch = 1;
    Blas::dense(input_size,
                output_size,
                batch,
                input.data(),
                weights.data(),
                output.data());
  
    if (ReLU) {
        for (auto o = int{0}; o < output_size; ++o) {
            output[o] = lambda_ReLU(biases[o] + output[o]);
        }
    } else {
        for (auto o = int{0}; o < output_size; ++o) {
            output[o] = biases[o] + output[o];
        }
    }
    return output;
}


void Convolve1::Forward(const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &output) {

     Blas::fixed_gemm((int)output_channels,
                      spatial_size,
                      (int)input_channels,
                      1.0f,
                      weights.data(),
                      (int)input_channels,
                      input.data(),
                      spatial_size,
                      0.0f,
                      output.data(),
                      spatial_size);
}


void AddSpatialBias::Forward(const size_t channels,
                             std::vector<float> &input,
                             const std::vector<float> &biases) {
    
    float *input_ptr = input.data();
    for (auto c = size_t{0}; c < channels; ++c) {
        for (auto b = size_t{0}; b < spatial_size; b++) {
            *input_ptr += biases[c];
            input_ptr++;
        }
    }
}


void Batchnorm::Forward(const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &means,
                        const std::vector<float> &stddevs,
                        const float *const eltwise,
                        const bool ReLU) {

    const auto lambda_ReLU = [&](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    float *input_ptr = input.data();
    const float *res = eltwise;
    if (eltwise) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto mean = means[c];
            const auto scale_stddev = stddevs[c];

            for (auto b = size_t{0}; b < spatial_size; b++) {
                float value = *input_ptr;
                value = scale_stddev * (value - mean) + *res;
                *input_ptr = lambda_ReLU(value);

                input_ptr++;
                res++;
            }
        }
    } else {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto mean = means[c];
            const auto scale_stddev = stddevs[c];

            for (auto b = size_t{0}; b < spatial_size; b++) {
                float value = *input_ptr;
                value = scale_stddev * (value - mean);
                *input_ptr = lambda_ReLU(value);
                input_ptr++;
            }
        }
    }
}

void GlobalAvgPool::Forward(const size_t channels,
                            const std::vector<float> &input,
                            std::vector<float> &output) {

    const float *input_ptr = input.data();

    for (auto c = size_t{0}; c < channels; ++c) {
        float Sum = 0.0f;
        for (auto b = size_t{0}; b < spatial_size; ++b) {
            Sum += *input_ptr;
            input_ptr++;
        }

        const float Mean = Sum / (float)spatial_size;
        output[c] = Mean;
    }
}

void SEUnit::Forward(const size_t channels,
                     const size_t se_size,
                     std::vector<float> &input,
                     const std::vector<float> &residual,
                     const std::vector<float> &weights_w1,
                     const std::vector<float> &weights_b1,
                     const std::vector<float> &weights_w2,
                     const std::vector<float> &weights_b2) {

    using pooling = GlobalAvgPool;
    auto pool = std::vector<float>(2 * channels);
    auto fc_out = std::vector<float>(se_size);

    pooling::Forward(channels, input, pool);
    FullyConnect::Forward(channels, se_size, pool, weights_w1, weights_b1, fc_out, true);
    FullyConnect::Forward(se_size, 2*channels, fc_out, weights_w2, weights_b2, pool, false);

    SEProcess(channels, input, residual, pool);
}

void SEUnit::SEProcess(const size_t channels,
                       std::vector<float> &input,
                       const std::vector<float> &residual,
                       const std::vector<float> &scale) {

    const auto lambda_ReLU = [](const auto val) {
        return (val > 0.0f) ? val : 0;
    };

    const auto lambda_sigmoid = [](const auto val) {
        return 1.0f / (1.0f + std::exp(-val));
    };

    auto gamma_ptr = scale.data();
    auto beta_ptr = scale.data() + channels;
    auto input_ptr = input.data();
    auto res_ptr = residual.data();


    for (auto c = size_t{0}; c < channels; ++c) {
        const auto gamma = lambda_sigmoid(*gamma_ptr);
        const auto beta = *beta_ptr;

        gamma_ptr++;
        beta_ptr++;

        for (auto i = size_t{0}; i < spatial_size; ++i) {
            float value = *input_ptr;
            *input_ptr = lambda_ReLU(gamma * value + beta + *res_ptr);
            input_ptr++;
            res_ptr++;
        }
    }
}


std::vector<float> Activation::Softmax(const std::vector<float> &input,
                                       const float temperature) {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(std::begin(input), std::end(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.emplace_back(val);
    }

    for (auto &out : output) {
        out /= denom;
    }

    return output;
}


std::vector<float> Activation::Tanh (const std::vector<float> &input) {

    auto output = std::vector<float>{};
    output.reserve(input.size());

    for (const auto &v : input) {
        output.emplace_back(std::tanh(v));
    }
    return output;
}

std::vector<float> Activation::Sigmoid(const std::vector<float> &input) {

    const auto lambda_sigmoid = [](const auto val) {
        return 1.0f / (1.0f + std::exp(-val));
    };

    auto output = std::vector<float>{};
    output.reserve(input.size());

    for (const auto &v : input) {
        output.emplace_back(lambda_sigmoid(v));
    }

    return output;
}
