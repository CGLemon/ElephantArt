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

#ifndef MODEL_H_INCLUDE
#define MODEL_H_INCLUDE

#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Position.h"
#include "Utils.h"

static constexpr auto INPUT_FEATURES = 4;
static constexpr auto INPUT_STATUS = 4;
static constexpr auto INPUT_MOVES = 1;
static constexpr auto INPUT_CHANNELS = INPUT_MOVES * 14 + INPUT_STATUS;

static constexpr auto POLICYMAP = 50;
static constexpr auto VALUELAYER = 256;
static constexpr auto WINRATELAYER = 4;

struct NNResult {
    std::array<float, POLICYMAP * Board::INTERSECTIONS> policy;
    std::array<float, WINRATELAYER> winrate_misc;
    NNResult () {
        policy.fill(0.0f);
        winrate_misc.fill(0.0f);
    }
};

struct Desc {
    struct ConvLayer {
        void load_weights(std::vector<float> &loadweights);
        void load_biases(std::vector<float> &loadweights);
        void load_size(int ic, int oc, int ks, bool check = true);
        
        int in_channels;
        int out_channels;
        int kernel_size;
        std::vector<float> weights;
        std::vector<float> biases;
    };

    struct BatchNormLayer {
        void load_means(std::vector<float> &loadweights);
        void load_stddevs(std::vector<float> &loadweights);
        void load_size(int c, bool check = true);
        
        int channels;
        std::vector<float> means;
        std::vector<float> stddevs;
    };

    struct LinearLayer {
        void load_weights(std::vector<float> &loadweights);
        void load_biases(std::vector<float> &loadweights);
        void load_size(int is, int os, bool check = true);
        
        int in_size;
        int out_size;
        std::vector<float> weights;
        std::vector<float> biases;
    };
};

struct Model {
    struct NNWeights {
        struct ResidualBlock{
            Desc::ConvLayer conv_1;
            Desc::BatchNormLayer bn_1;
            Desc::ConvLayer conv_2;
            Desc::BatchNormLayer bn_2;

            Desc::LinearLayer extend;
            Desc::LinearLayer squeeze;
            int se_size{0};
            bool apply_se;
        };
        bool loaded{false};
        bool winograd{false};
        int input_channels{0};
        int input_features{0};
        int residual_blocks{0};
        int residual_channels{0};
        int policy_extract_channels{0};
        int policy_map{0};
        int value_extract_channels{0};

        // input layer
        Desc::ConvLayer input_conv;
        Desc::BatchNormLayer input_bn;
        Desc::LinearLayer input_fc1;
        Desc::LinearLayer input_fc2;

        // residual tower
        std::vector<ResidualBlock> residual_tower;

        // policy head
        Desc::ConvLayer p_ex_conv;
        Desc::BatchNormLayer p_ex_bn;

        Desc::ConvLayer p_map;

        // value head
        Desc::ConvLayer v_ex_conv;
        Desc::BatchNormLayer v_ex_bn;
        Desc::LinearLayer v_fc1;
        Desc::LinearLayer v_fc2;
    };

    class NNPipe {
    public:
        virtual void initialize(std::shared_ptr<NNWeights> weights) = 0;
        virtual void forward(const std::vector<float> &planes,
                             const std::vector<float> &features,
                             std::vector<float> &output_pol,
                             std::vector<float> &output_val) = 0;

        virtual void reload(std::shared_ptr<Model::NNWeights> weights) = 0;
        virtual void release() = 0;
        
        virtual void destroy() = 0;
        virtual bool valid() = 0;
    };
    
    static std::vector<float> gather_planes(const Position *const position,
                                            const int symmetry);
    static std::vector<float> gather_features(const Position *const position);

    static void load_weights(const std::string &filename,
                             std::shared_ptr<NNWeights> &nn_weight);
    
    static void process_weights(std::shared_ptr<NNWeights> &nn_weight);

    static void dump_nn_info(std::shared_ptr<NNWeights> &nn_weight, Utils::Timer &timer);

    static void fill_weights(std::istream &weights_file,
                             std::shared_ptr<NNWeights> &nn_weight);
    
    static void fill_fullyconnect_layer(Desc::LinearLayer &layer,
                                        std::istream &weights_file,
                                        const int in_size,
                                        const int out_size);

    static void fill_batchnorm_layer(Desc::BatchNormLayer &layer,
                                     std::istream &weights_file,
                                     const int channels);

    static void fill_convolution_layer(Desc::ConvLayer &layer,
                                       std::istream &weights_file,
                                       const int in_channels,
                                       const int out_channels,
                                       const int kernel_size);

    static NNResult get_result(std::vector<float> &policy,
                               std::vector<float> &value,
                               const float p_softmax_temp,
                               const float v_softmax_temp,
                               const int symmetry);
};
#endif
