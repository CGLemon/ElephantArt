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

#include "WinogradHelper.h"
#include "Blas.h"

std::vector<float> Winograd::transform_f(const std::vector<float> &f,
                                         const int outputs,
                                         const int channels) {
    // F(4x4, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    const auto G = std::array<float, KERNEL_SIZE * WINOGRAD_ALPHA>
                    { 1.0f,        0.0f,      0.0f,
                      -2.0f/3.0f, -SQ2/3.0f, -1.0f/3.0f,
                      -2.0f/3.0f,  SQ2/3.0f, -1.0f/3.0f,
                      1.0f/6.0f,   SQ2/6.0f,  1.0f/3.0f,
                      1.0f/6.0f,  -SQ2/6.0f,  1.0f/3.0f,
                      0.0f,        0.0f,      1.0f};

    auto temp = std::array<float, KERNEL_SIZE * WINOGRAD_ALPHA>{};

    constexpr auto max_buffersize = 8;
    auto buffersize = max_buffersize;

    if (outputs % buffersize != 0) {
        buffersize = 1;
    }

    auto buffer = std::array<float, max_buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA>{};

    for (auto c = 0; c < channels; c++) {
        for (auto o_b = 0; o_b < outputs/buffersize; o_b++) {
            for (auto bufferline = 0; bufferline < buffersize; bufferline++) {
                const auto o = o_b * buffersize + bufferline;
                const auto f_offset = o * channels * FILTER_LEN + c * FILTER_LEN;
                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < KERNEL_SIZE; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < KERNEL_SIZE; k++) {
                            acc += G[i*KERNEL_SIZE + k] * f[f_offset + k*KERNEL_SIZE + j];
                        }
                        temp[i*KERNEL_SIZE + j] = acc;
                    }
                }

                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < KERNEL_SIZE; k++) {
                            acc += temp[xi*KERNEL_SIZE + k] * G[nu*KERNEL_SIZE + k];
                        }
                        buffer[(xi * WINOGRAD_ALPHA + nu) * buffersize + bufferline] = acc;
                    }
                }
            }
            for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                for (auto entry = 0; entry < buffersize; entry++) {
                    const auto o = o_b * buffersize + entry;
                    U[i * outputs * channels
                      + c * outputs
                      + o] =
                    buffer[buffersize * i + entry];
                }
            }
        }
    }

    U.shrink_to_fit();
    return U;
}

void Winograd::transform_in(const std::vector<float> &in,
                            std::vector<float> &V, const int C) {

    static constexpr auto Wpad_X = 2 + WINOGRAD_M * WTILES_X;
    static constexpr auto Wpad_Y = 2 + WINOGRAD_M * WTILES_Y;
    static constexpr auto buffersize = 32;

    std::array<std::array<float, Wpad_Y>, Wpad_X> in_pad{0.0f};
    std::array<float, buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;
    int buffer_offset = 0;
    int buffer_entries = 0;

    // multiple vector [i0..i5] by Bt and produce [o0..o5]
    // const auto Bt = std::array<float, WINOGRAD_TILE>
    //           {1.0f,  0.0f,     -5.0f/2.0f,  0.0f,      1.0f, 0.0f,
    //            0.0f, -SQ2,      -2.0f,       SQ2/2.0f,  1.0f, 0.0f,
    //            0.0f,  SQ2,      -2.0f,      -SQ2/2.0f,  1.0f, 0.0f,
    //            0.0f, -SQ2/2.0f, -1.0f/2.0f,  SQ2,       1.0f, 0.0f,
    //            0.0f,  SQ2/2.0f, -1.0f/2.0f, -SQ2,       1.0f, 0.0f,
    //            0.0f,  1.0f,      0.0f,      -5.0f/2.0f, 0.0f, 1.0f};
    const auto multiply_bt = [](float &o0, float &o1, float &o2, float &o3, float &o4,
                                float &o5, float i0, float i1, float i2, float i3,
                                float i4, float i5) {
        auto i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
        auto i4m2 = i2 * -2.0f + i4 * 1.0f;

        o0 = i0 + i2 * (-5.0f / 2.0f) + i4;
        o1 = i3m1 + i4m2;
        o2 = -i3m1 + i4m2;

        auto i3m1_2 = i3 * (SQ2) + i1 * (-SQ2 / 2.0f);
        auto i4m2_2 = i2 * (-1.0f / 2.0f) + i4;

        o3 = i3m1_2 + i4m2_2;
        o4 = -i3m1_2 + i4m2_2;

        o5 = i1 + i3 * (-5.0f / 2.0f) + i5;
    };

    for (auto ch = 0; ch < C; ch++) {
        for (auto yin = 0; yin < H; yin++) {
            for (auto xin = 0; xin < W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch * (W * H) + yin * W + xin];
            }
        }
        for (auto block_y = 0; block_y < WTILES_Y; block_y++) {
            // Tiles overlap by 2
            const auto yin = WINOGRAD_M * block_y;
            for (auto block_x = 0; block_x < WTILES_X; block_x++) {
                const auto xin = WINOGRAD_M * block_x;
#define DECL_T1(XX)                                                            \
  float T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4,       \
      T1_##XX##_5;
                DECL_T1(0)
                DECL_T1(1)
                DECL_T1(2)
                DECL_T1(3)
                DECL_T1(4)
                DECL_T1(5)

              // Calculates transpose(B).x.B
#define MULTIPLY_BT(XX)                                                        \
  multiply_bt(T1_0_##XX, T1_1_##XX, T1_2_##XX, T1_3_##XX, T1_4_##XX,           \
              T1_5_##XX, in_pad[yin + 0][xin + XX], in_pad[yin + 1][xin + XX], \
              in_pad[yin + 2][xin + XX], in_pad[yin + 3][xin + XX],            \
              in_pad[yin + 4][xin + XX], in_pad[yin + 5][xin + XX]);
                MULTIPLY_BT(0)
                MULTIPLY_BT(1)
                MULTIPLY_BT(2)
                MULTIPLY_BT(3)
                MULTIPLY_BT(4)
                MULTIPLY_BT(5)

#define MULTIPLY_B(XX)                                                         \
  multiply_bt(buffer[buffersize * (XX * WINOGRAD_ALPHA + 0) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 1) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 2) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 3) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 4) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 5) + buffer_entries], \
              T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4, \
              T1_##XX##_5);
                MULTIPLY_B(0)
                MULTIPLY_B(1)
                MULTIPLY_B(2)
                MULTIPLY_B(3)
                MULTIPLY_B(4)
                MULTIPLY_B(5)

                if (buffer_entries == 0) {
                    buffer_offset = ch * P + block_y * WTILES_X + block_x;
                }
                buffer_entries++;

                if (buffer_entries >= buffersize ||
                    (ch == C - 1 && block_x == WTILES_X - 1 && block_y == WTILES_Y - 1)) {

                    for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                        for (auto entry = 0; entry < buffer_entries; entry++) {
                            V[i * C * P + buffer_offset + entry] =
                                buffer[i * buffersize + entry];
                        }
                    }
                    buffer_entries = 0;
                }
            }
        }
    }
}

void Winograd::sgemm(const std::vector<float> &U,
                     const std::vector<float> &V,
                     std::vector<float> &M, const int C,
                     const int K) {

    for (int b = 0; b < WINOGRAD_TILE; b++) {
        const auto u_offset = b * K * C;
        const auto v_offset = b * C * P;
        const auto m_offset = b * K * P;

        Blas::winograd_gemm(u_offset,
                            v_offset,
                            m_offset,
                            K,
                            P,
                            C,
                            1.0f,
                            U.data(),
                            K,
                            V.data(),
                            P,
                            0.0f,
                            M.data(),
                            P);
    }
}

void Winograd::transform_out(const std::vector<float> &M,
                             std::vector<float> &Y, const int K) {

    // multiple vector [i0..i5] by At and produce [o0..o3]
    // const auto At = std::array<float, WINOGRAD_ALPHA * WINOGRAD_M>
    //       {1.0f, 1.0f,      1.0f,       1.0f,      1.0f,     0.0f,
    //        0.0f, SQ2/2.0f, -SQ2/2.0f,   SQ2,      -SQ2,      0.0f,
    //        0.0f, 1.0f/2.0f, 1.0f/2.0f,  2.0f,      2.0f,     0.0f,
    //        0.0f, SQ2/4.0f, -SQ2/4.0f,   2.0f*SQ2, -2.0f*SQ2, 1.0f};
    const auto multiply_at = [](float &o0, float &o1, float &o2, float &o3, float i0,
                                float i1, float i2, float i3, float i4, float i5) {
        auto t1p2 = (i1 + i2) * (1.0f / 2.0f);
        auto t1m2 = (i1 - i2) * (SQ2 / 4.0f);
        auto t3p4 = i3 + i4;
        auto t3m4 = (i3 - i4) * (SQ2);

        o0 = i0 + t1p2 + t1p2 + t3p4;
        o1 = t1m2 + t1m2 + t3m4;
        o2 = t1p2 + t3p4 + t3p4;
        o3 = t1m2 + t3m4 + t3m4 + i5;
    };

    for (auto k = 0; k < K; k++) {
        for (auto block_x = 0; block_x < WTILES_X; block_x++) {
            const auto x = WINOGRAD_M * block_x;
            for (auto block_y = 0; block_y < WTILES_Y; block_y++) {
                const auto y = WINOGRAD_M * block_y;

                const auto b = block_y * WTILES_X + block_x;
                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_ALPHA>;

                auto temp_m = WinogradTile{};
                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        temp_m[xi][nu] = M[(xi * WINOGRAD_ALPHA + nu) * K * P + k * P + b];
                    }
                }
                auto temp = std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_M>{};
                auto o = std::array<std::array<float, WINOGRAD_M>, WINOGRAD_M>{};

                // Calculates transpose(A).temp_m.A
                for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                    multiply_at(temp[0][j], temp[1][j], temp[2][j], temp[3][j],
                                temp_m[0][j], temp_m[1][j], temp_m[2][j], temp_m[3][j],
                                temp_m[4][j], temp_m[5][j]);
                }

                for (auto i = 0; i < WINOGRAD_M; i++) {
                    multiply_at(o[i][0], o[i][1], o[i][2], o[i][3], temp[i][0],
                                temp[i][1], temp[i][2], temp[i][3], temp[i][4],
                                temp[i][5]);
                }

                const auto y_ind = k * H * W + y * W + x;
                for (auto i = 0; i < WINOGRAD_M; i++) {
                    for (auto j = 0; j < WINOGRAD_M; j++) {
                        if (y + i < H && x + j < W) {
                            Y[y_ind + i * W + j] = o[i][j];
                        }
                    }
                }
            }
        }
    }
}

void Winograd::Forward(const size_t input_channels,
                       const size_t output_channels,
                       const std::vector<float> &input,
                       const std::vector<float> &U,
                       std::vector<float> &V,
                       std::vector<float> &M,
                       std::vector<float> &output) {

    transform_in(input, V, input_channels);
    sgemm(U, V, M, input_channels, output_channels);
    transform_out(M, output, output_channels);
}

std::pair<size_t, size_t> Winograd::get_workspace_size(const size_t input_channels,
                                                       const size_t output_channels) {

    auto winograd_V_size = WINOGRAD_TILE * input_channels * P;
    auto winograd_M_size = WINOGRAD_TILE * output_channels * P;
    return std::make_pair(winograd_V_size, winograd_M_size);
}
