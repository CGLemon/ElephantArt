#ifndef WINOGRAD_HELPER_H_INCLUDE
#define WINOGRAD_HELPER_H_INCLUDE
#include <vector>
#include <array>
#include "config.h"
#include "Board.h"

static constexpr auto WINOGRAD_M = 4;
static constexpr auto WINOGRAD_ALPHA = WINOGRAD_M + 3 - 1; // 6
static constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA; // 36
static constexpr auto SQ2 = 1.4142135623730951f; // Square root of 2

class Winograd {
public:
    static std::vector<float> transform_f(const std::vector<float> &f,
                                          const int outputs,
                                          const int channels);

    static std::pair<size_t, size_t> get_workspace_size(const size_t input_channels,
                                                        const size_t output_channels);


    void transform_in(const std::vector<float> &in,
                      std::vector<float> &V, const int C);

    void sgemm(const std::vector<float> &U,
               const std::vector<float> &V,
               std::vector<float> &M, const int C,
               const int K);

    void transform_out(const std::vector<float> &M,
                       std::vector<float> &Y, const int K);

    void Forward(const size_t input_channels,
                 const size_t output_channels,
                 const std::vector<float> &input,
                 const std::vector<float> &U,
                 std::vector<float> &V,
                 std::vector<float> &M,
                 std::vector<float> &output);

private:
    static constexpr auto KERNEL_SIZE = 3;
    static constexpr auto FILTER_LEN = KERNEL_SIZE * KERNEL_SIZE;
    static constexpr auto W = Board::WIDTH;
    static constexpr auto H = Board::HEIGHT;
    static constexpr auto WTILES_X = (W / WINOGRAD_M + (W % WINOGRAD_M != 0));
    static constexpr auto WTILES_Y = (H / WINOGRAD_M + (H % WINOGRAD_M != 0));
    static constexpr auto P = WTILES_X * WTILES_Y;
};


#endif

