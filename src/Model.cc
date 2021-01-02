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

#include <cassert>
#include <unordered_map>

#include "Blas.h"
#include "Model.h"
#include "Utils.h"
#include "config.h"
#include "WinogradHelper.h"

#ifdef USE_FAST_PARSER
#include "fast_double_parser.h"
#endif 

template <typename container>
void process_bn_var(container &weights) {
    static constexpr float epsilon = 1e-5f;
    for (auto &&w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

void Desc::ConvLayer::load_weights(std::vector<float> &loadweights) {
    weights = std::move(loadweights);
    weights.shrink_to_fit();
}

void Desc::ConvLayer::load_biases(std::vector<float> &loadweights) {
    biases = std::move(loadweights);
    biases.shrink_to_fit();
}

void Desc::ConvLayer::load_size(int ic, int oc, int ks, bool check) {
    in_channels = ic;
    out_channels = oc;
    kernel_size = ks;
    if (check && in_channels * out_channels * kernel_size * kernel_size != (int)weights.size()) {
        throw "The one of Convolutional Layers weights size is not acceptable";
    }
    if (check && out_channels != (int)biases.size()) {
        throw "The one of Convolutional Layers baies size is not acceptable";
    }
}

void Desc::BatchNormLayer::load_means(std::vector<float> &loadweights) {
    means = std::move(loadweights);
    means.shrink_to_fit();
}

void Desc::BatchNormLayer::load_stddevs(std::vector<float> &loadweights) {
    process_bn_var(loadweights);
    stddevs = std::move(loadweights);
    stddevs.shrink_to_fit();
}

void Desc::BatchNormLayer::load_size(int c, bool check) {
    channels = c;
    if (check && channels != (int)stddevs.size()) {
        throw "The one of BatchNorm Layers stddevs size is not acceptable";
    }
    if (check && channels != (int)means.size()) {
        throw "The one of BatchNorm Layers means size is not acceptable";
    }
}

void Desc::LinearLayer::load_weights(std::vector<float> &loadweights) {
    weights = std::move(loadweights);
    weights.shrink_to_fit();
}

void Desc::LinearLayer::load_biases(std::vector<float> &loadweights) {
    biases = std::move(loadweights);
    biases.shrink_to_fit();
}

void Desc::LinearLayer::load_size(int is, int os, bool check) {
    in_size = is;
    out_size = os;
    if (check && in_size * out_size != (int)weights.size()) {
        throw "The one of FullyConnect Layers weights size is not acceptable";
    }
    if (check && out_size != (int)biases.size()) {
        throw "The one of FullyConnect Layers baies size is not acceptable";
    }
}

void fill_piece_planes(const std::shared_ptr<const Board> board,
                       std::vector<float>::iterator red,
                       std::vector<float>::iterator black,
                       const int symmetry) {
    
    for (auto idx = size_t{0}; idx < Board::INTERSECTIONS; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        const auto sym_x = sym_idx % Board::WIDTH;
        const auto sym_y = sym_idx / Board::WIDTH;
        const auto sym_vtx = Board::get_vertex(sym_x, sym_y);
        const auto pis = board->get_piece(sym_vtx);
        
        if (static_cast<int>(pis) <  7) {
            red[static_cast<int>(pis) * Board::INTERSECTIONS + idx] = static_cast<float>(true);
        } else if (static_cast<int>(pis) < 14) {
            black[(static_cast<int>(pis)-7) * Board::INTERSECTIONS + idx] = static_cast<float>(true);
        }
        assert(pis != Types::INVAL_PIECE);
    }
}

std::vector<float> Model::gather_planes(const Position *const position,
                                        const int symmetry) {

    constexpr auto MOVES_PLANES = INPUT_MOVES * 14;
    constexpr auto STATUS_PLANES = INPUT_STATUS;
    static_assert(INPUT_CHANNELS == MOVES_PLANES + STATUS_PLANES, "");
    static_assert(INPUT_CHANNELS == 18, "");

    // planes |  1 -  7 | current player picee position.
    // planes |  8      | current player is red or not.
    // planes |  9 - 15 | next player picee position.
    // planes | 16      | next player is red or not.
    // planes | 17 - 18 | repeat conut

    auto input_data = std::vector<float>(INPUT_CHANNELS * Board::INTERSECTIONS, 0.0f);
    auto color = position->get_to_move();
    auto blk_iterator = std::begin(input_data);
    auto red_iterator = std::begin(input_data);
    if (color == Types::BLACK) {
        std::advance(red_iterator, (1 + INPUT_MOVES * 7) * Board::INTERSECTIONS);
    } else {
        std::advance(blk_iterator, (1 + INPUT_MOVES * 7) * Board::INTERSECTIONS);
    }

    const auto movenum = position->get_movenum();
    const auto past_moves = std::min(INPUT_MOVES-1, movenum);
    
    // plane 1-7 and 9-15
    for (auto p = 0; p < INPUT_MOVES; ++p) {
        if (p < past_moves) {
            const auto board = position->get_past_board(p);
            fill_piece_planes(board,
                              red_iterator,
                              blk_iterator,
                              symmetry);
        }
        std::advance(red_iterator, 7 * Board::INTERSECTIONS);
        std::advance(blk_iterator, 7 * Board::INTERSECTIONS);
    }

    // plane 8 or 16
    std::fill(red_iterator, red_iterator + Board::INTERSECTIONS, static_cast<float>(true));

    // plane 17 - 18
    // Not complete yet

    return input_data;
}

void Model::load_weights(const std::string &filename,
                         std::shared_ptr<NNWeights> &nn_weight) {
    auto file = std::ifstream{};
    auto buffer = std::stringstream{};
    auto line = std::string{};

    file.open(filename.c_str());

    if (!file.is_open()) {
        file.close();
        Utils::auto_printf("Could not opne file : %s!\n", filename.c_str());
        return;
    }

    while(std::getline(file, line)) {
        buffer << line;
        buffer << std::endl;
    }
    file.close();
    
    try {
        fill_weights(buffer, nn_weight);
    } catch (const char* err) {
        Utils::auto_printf("Loading network file warning!\n", err);
        Utils::auto_printf("    Cause : %s.\n", err);
    }
    
    if (nn_weight->loaded) {
        Utils::auto_printf("Loading is successful!\n");
    }
}


void Model::fill_weights(std::istream &weights_file,
                         std::shared_ptr<NNWeights> &nn_weight) {
    auto timer = Utils::Timer{};
    auto counter = size_t{0};

    // Part 1.
    auto line = std::string{};
    if (std::getline(weights_file, line)) {
        const auto p = Utils::CommandParser(line);
        if (p.get_command(0)->str != "fork" ||
                p.get_command(1)->str != "main") {
            throw "Weights file format is not acceptable";
        } else {
            counter++;
        }
    } else {
        throw "Weights file is empty";
    }

    // Part 2.
    using LayerShape = std::vector<int>;
    using NetModel = std::vector<LayerShape>;
    using NetInfo = std::unordered_map<std::string, std::string>;
    
    const auto get_netinfo = [](NetInfo &netinfo,
                                std::istream &weights_file) -> void {
        auto line = std::string{};
        while (std::getline(weights_file, line)) {
            const auto p = Utils::CommandParser(line);
            if (p.get_command(0)->str[0] == '#') {
                continue;
            } else if (p.get_command(0)->str == "end") {
                break;
            }
            netinfo.emplace(p.get_command(0)->str, p.get_command(1)->str);
        }
    };
    
    const auto get_netmode = [](NetModel &netmodel,
                                std::istream &weights_file) -> void {
        auto line = std::string{};
        auto cnt = size_t{0};
        while (std::getline(weights_file, line)) {
            const auto p = Utils::CommandParser(line);
            if (p.get_command(0)->str[0] == '#') {
                continue;
            } else if (p.get_command(0)->str == "end") {
                break;
            }
            
            netmodel.emplace_back(LayerShape{});
            for (auto i = size_t{1}; i < p.get_count(); ++i) {
                const auto s = p.get_command(i)->get<int>();
                netmodel[cnt].emplace_back(s);
            }

            if (p.get_command(0)->str == "FullyConnect") {
                if (netmodel[cnt].size() != 2) {
                    throw "The FullyConnect Layer shape is error";
                }
            } else if (p.get_command(0)->str == "Convolve") {
                if (netmodel[cnt].size() != 3) {
                    throw "The Convolve Layer shape is error";
                }
            } else if (p.get_command(0)->str == "BatchNorm") {
                if (netmodel[cnt].size() != 1) {
                    throw "The BatchNorm layer shape is error";
                }
            } else {
                throw "The layer shape is error";
            }
            cnt++;
        }
    };

    const auto fill_layer_weights = [&](NetInfo &netinfo,
                                        NetModel &netmodel,
                                        std::istream &weights_file,
                                        std::shared_ptr<NNWeights> &nn_weight) -> void {
        
        nn_weight->residual_blocks = std::stoi(netinfo["ResidualBlocks"]);
        nn_weight->residual_channels = std::stoi(netinfo["ResidualChannels"]);
        nn_weight->policy_extract_channels = std::stoi(netinfo["PolicyExtract"]);
        nn_weight->value_extract_channels = std::stoi(netinfo["ValueExtract"]);
        
        nn_weight->input_channels = std::stoi(netinfo["InputChannels"]);
        nn_weight->policy_map = std::stoi(netinfo["PolicyMap"]);
        
        if (nn_weight->input_channels != INPUT_CHANNELS) {
            throw "The number of input channels is wrong.";
        }
        if  (nn_weight->policy_map != POLICYMAP) {
            throw "The number of policy map channels is wrong.";
        }
        
        auto se_cnt = 0;
        // input layer
        const auto input_conv_shape = netmodel[0];
        fill_convolution_layer(nn_weight->input_conv,
                               weights_file,
                               input_conv_shape[0],
                               input_conv_shape[1],
                               input_conv_shape[2]);
        
        const auto input_bn_shape = netmodel[1];
        fill_batchnorm_layer(nn_weight->input_bn,
                             weights_file,
                             input_bn_shape[0]);
        
        const auto residuals = nn_weight->residual_blocks;

        if (nn_weight->residual_channels != input_conv_shape[1] ||
            nn_weight->residual_channels != input_bn_shape[0] || 
            input_conv_shape[2] != 3) {
            throw "The input layer is wrong";
        }
        timer.record();
        // residual tower
        for (int b = 0; b < residuals; ++b) {

            const auto off_set = b * 4 + 2 + 2 * se_cnt;
            const auto res_conv1_shape = netmodel[off_set];
            const auto res_bn1_shape = netmodel[off_set+1];
            const auto res_conv2_shape = netmodel[off_set+2];
            const auto res_bn2_shape = netmodel[off_set+3];
            
            nn_weight->residual_tower.emplace_back(NNWeights::ResidualBlock{});
            auto tower_ptr = nn_weight->residual_tower.data() + b;
        
            fill_convolution_layer(tower_ptr->conv_1,
                                   weights_file,
                                   res_conv1_shape[0],
                                   res_conv1_shape[1],
                                   res_conv1_shape[2]);

            fill_batchnorm_layer(tower_ptr->bn_1,
                                 weights_file,
                                 res_bn1_shape[0]);

            if (nn_weight->residual_channels != res_conv1_shape[0] ||
                nn_weight->residual_channels != res_conv1_shape[1] ||
                nn_weight->residual_channels != res_bn1_shape[0] || 
                res_conv1_shape[2] != 3) {
                throw "The Residual Block (1) is wrong";
            }

            fill_convolution_layer(tower_ptr->conv_2,
                                   weights_file,
                                   res_conv2_shape[0],
                                   res_conv2_shape[1],
                                   res_conv2_shape[2]);

            fill_batchnorm_layer(tower_ptr->bn_2,
                                 weights_file,
                                 res_bn2_shape[0]);

            if (nn_weight->residual_channels != res_conv2_shape[0] ||
                nn_weight->residual_channels != res_conv2_shape[1] ||
                nn_weight->residual_channels != res_bn2_shape[0] ||
                res_conv2_shape[2] != 3) {
                throw "The Residual Block (2) is wrong";
            }
            
            const auto res_next_shape = netmodel[off_set+4];
            
            if (res_next_shape.size() == 2 /* fullyconnect layer */) {

                tower_ptr->apply_se = true;
                se_cnt++;
                const auto se_extend_shape = netmodel[off_set+4];
                const auto se_squeeze_shape = netmodel[off_set+5];
                fill_fullyconnect_layer(tower_ptr->extend,
                                        weights_file,
                                        se_extend_shape[0],
                                        se_extend_shape[1]);
                fill_fullyconnect_layer(tower_ptr->squeeze,
                                        weights_file,
                                        se_squeeze_shape[0],
                                        se_squeeze_shape[1]);

                if (se_extend_shape[1] != se_squeeze_shape[0]) {
                    throw "The SE Unit size is wrong.";
                }
                if (2 * se_extend_shape[0] != se_squeeze_shape[1] ||
                    se_extend_shape[0] != nn_weight->residual_channels) {
                    throw "The SE Unit size is wrong.";
                }
                
                tower_ptr->se_size = se_extend_shape[1];
            } else {
                tower_ptr->apply_se = false;
            }
        }

        timer.record();
        const auto off_set = residuals * 4 + 2 + 2 * se_cnt;

        // policy head
        const auto p_ex_conv_shape = netmodel[off_set];
        fill_convolution_layer(nn_weight->p_ex_conv,
                               weights_file,
                               p_ex_conv_shape[0],
                               p_ex_conv_shape[1],
                               p_ex_conv_shape[2]);
        
        const auto p_ex_bn_shape = netmodel[off_set+1];
        fill_batchnorm_layer(nn_weight->p_ex_bn,
                             weights_file,
                             p_ex_bn_shape[0]);
        
        const auto p_map_shape = netmodel[off_set+2];
        fill_convolution_layer(nn_weight->p_map,
                               weights_file,
                               p_map_shape[0],
                               p_map_shape[1],
                               p_map_shape[2]);

        if (p_ex_conv_shape[2] != 3 || p_map_shape[2] != 3) {
            throw "The policy map kernel size is wrong";
        }
        if (p_map_shape[1] != POLICYMAP) {
            throw "The number of policy map channels size is wrong";
        }
         
        // value head
        const auto v_ex_conv_shape = netmodel[off_set+3];
        fill_convolution_layer(nn_weight->v_ex_conv,
                               weights_file,
                               v_ex_conv_shape[0],
                               v_ex_conv_shape[1],
                               v_ex_conv_shape[2]);
        
        const auto v_ex_bn_shape = netmodel[off_set+4];
        fill_batchnorm_layer(nn_weight->v_ex_bn,
                             weights_file,
                             v_ex_bn_shape[0]);
        
        const auto v_fc1_shape = netmodel[off_set+5];
        fill_fullyconnect_layer(nn_weight->v_fc1,
                                weights_file,
                                v_fc1_shape[0],
                                v_fc1_shape[1]);
        
        const auto v_fc2_shape = netmodel[off_set+6];
        fill_fullyconnect_layer(nn_weight->v_fc2,
                                weights_file,
                                v_fc2_shape[0],
                                v_fc2_shape[1]);

        if (v_ex_bn_shape[0] != v_ex_conv_shape[1]) {
            throw "";
        }
        if (v_ex_conv_shape[2] != 1) {
            throw "The value layer kernel size is wrong";
        }
        if (v_fc1_shape[0] != v_ex_conv_shape[1] * Board::INTERSECTIONS ||
            v_fc1_shape[1] != VALUELAYER ||
            v_fc2_shape[0] != v_fc1_shape[1] ||
            v_fc2_shape[1] != WINRATELAYER) {
            throw "The value layer size is wrong";
        }

        std::getline(weights_file, line);
        const auto p = Utils::CommandParser(line);
        if (p.get_command(0)->str != "end") {
            throw "Not end? Weights file format is not acceptable";
        }

        nn_weight->loaded = true;
        timer.record();

        process_weights(nn_weight);
    };


    auto netinfo = NetInfo{};
    auto netmodel = NetModel{};
    while (std::getline(weights_file, line)) {
        const auto p = Utils::CommandParser(line, 2);
        if (p.get_command(0)->str == "fork") {
            if (p.get_command(1)->str == "info") {
                get_netinfo(netinfo, weights_file);
                auto nntype = netinfo["NNType"];
                if (nntype != "Residual") {
                    throw "Only support Residual Network";
                }
            } else if (p.get_command(1)->str == "model") {
                get_netmode(netmodel, weights_file);
            } else {
                counter++;
            }
        } else if (p.get_command(0)->str == "end") {
            counter--;
        }
    }

    if (counter != 0) {
        throw "Weights file format is not acceptable";
    }
    
    if (netinfo.empty() || netmodel.empty()) {
        throw "The weighs information must be provided";
    }
    timer.record();

    weights_file.clear();
    weights_file.seekg(0, std::ios::beg);
    while (std::getline(weights_file, line)) {
        const auto p = Utils::CommandParser(line);
        if (p.get_command(0)->str == "fork") {
            if (p.get_command(1)->str == "parameters") {
                fill_layer_weights(netinfo,
                                   netmodel,
                                   weights_file,
                                   nn_weight);
            }
        }
    }

    dump_nn_info(nn_weight, timer);
}

NNResult Model::get_result(std::vector<float> &policy,
                           std::vector<float> &value,
                           const float p_softmax_temp,
                           const float v_softmax_temp,
                           const int symmetry) {
    NNResult result;

    // Probabilities
    const auto probabilities = Activation::Softmax(policy, p_softmax_temp);
    for (auto p = size_t{0}; p < POLICYMAP; ++p) {
        for (auto idx = size_t{0}; idx < Board::INTERSECTIONS; ++idx) {
            const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
            result.policy[sym_idx + p * Board::INTERSECTIONS] =
                probabilities[idx + p * Board::INTERSECTIONS];
        }
    }

    // Winrate
    const auto wdl_raw = std::vector<float>{value[0], value[1], value[2]}; 
    const auto wdl = Activation::Softmax(wdl_raw, v_softmax_temp);

    result.winrate_misc[0] = wdl[0];                  // wdl head win probability
    result.winrate_misc[1] = wdl[1];                  // wdl head draw probability
    result.winrate_misc[2] = wdl[2];                  // wdl head loss probability
    result.winrate_misc[3] = std::tanh(value[3]);     // stm head winrate

    return result;
}

void Model::process_weights(std::shared_ptr<NNWeights> &nn_weight) {
    // input layer
    for (auto idx = size_t{0}; idx < nn_weight->input_conv.biases.size(); ++idx) {
        nn_weight->input_bn.means[idx] -= nn_weight->input_conv.biases[idx] *
                                              nn_weight->input_bn.stddevs[idx];
        nn_weight->input_conv.biases[idx] = 0.0f;
    }
    // residual tower
    for (auto &residual : nn_weight->residual_tower) {
        for (auto idx = size_t{0}; idx < residual.conv_1.biases.size(); ++idx) {
            residual.bn_1.means[idx] -= residual.conv_1.biases[idx] *
                                            residual.bn_1.stddevs[idx];
            residual.conv_1.biases[idx] = 0.0f;
        }
        for (auto idx = size_t{0}; idx < residual.conv_2.biases.size(); ++idx) {
            residual.bn_2.means[idx] -= residual.conv_2.biases[idx] *
                                            residual.bn_2.stddevs[idx];
            residual.conv_2.biases[idx] = 0.0f;
        }
    }
    // policy head
    for (auto idx = size_t{0}; idx < nn_weight->p_ex_conv.biases.size(); ++idx) {
        nn_weight->p_ex_bn.means[idx] -= nn_weight->p_ex_conv.biases[idx] *
                                             nn_weight->p_ex_bn.stddevs[idx];
        nn_weight->p_ex_conv.biases[idx] = 0.0f;
    }
    // value head
    for (auto idx = size_t{0}; idx < nn_weight->v_ex_conv.biases.size(); ++idx) {
        nn_weight->v_ex_bn.means[idx] -= nn_weight->v_ex_conv.biases[idx] *
                                             nn_weight->v_ex_bn.stddevs[idx];
        nn_weight->v_ex_conv.biases[idx] = 0.0f;
    }

    if (option<bool>("winograd")) {
        nn_weight->winograd = true;
    } else {
        return;
    }
    auto channels = nn_weight->residual_channels;
    if (nn_weight->input_conv.kernel_size == 3) {
        nn_weight->input_conv.weights = Winograd::transform_f(
                                            nn_weight->input_conv.weights,
                                            channels, INPUT_CHANNELS);
    }
    for (auto &residual : nn_weight->residual_tower) {
        if (residual.conv_1.kernel_size == 3) {
            residual.conv_1.weights = Winograd::transform_f(
                                          residual.conv_1.weights,
                                          channels, channels);
        }
        if (residual.conv_2.kernel_size == 3) { 
            residual.conv_2.weights = Winograd::transform_f(
                                          residual.conv_2.weights,
                                          channels, channels);
        }
    }

    if (nn_weight->p_ex_conv.kernel_size == 3) {
        auto p_channels = nn_weight->policy_extract_channels;
        nn_weight->p_ex_conv.weights = Winograd::transform_f(
                                           nn_weight->p_ex_conv.weights,
                                           p_channels, channels);
    }

    if (nn_weight->p_map.kernel_size == 3) {
        auto p_channels = nn_weight->policy_extract_channels;
        nn_weight->p_map.weights = Winograd::transform_f(
                                       nn_weight->p_map.weights,
                                       POLICYMAP, p_channels);
    }

    if (nn_weight->v_ex_conv.kernel_size == 3) {
        auto v_channels = nn_weight->value_extract_channels;
        nn_weight->p_map.weights = Winograd::transform_f(
                                       nn_weight->v_ex_conv.weights,
                                       v_channels, channels);
    }

}

void Model::dump_nn_info(std::shared_ptr<NNWeights> &nn_weight, Utils::Timer &timer) {

    const auto duration = [](Utils::Timer &timer, int t) -> float {
        auto cnt = timer.get_record_count();
        if (t == 1) {
            return timer.get_record_time(1);
        } else if (t > 1 && t <= cnt) {
            return timer.get_record_time((size_t)t) - timer.get_record_time((size_t)t-1);
        }
        return 0;
    };

    Utils::auto_printf("Neural Network Information :\n");
    Utils::auto_printf("Time :\n");
    Utils::auto_printf("  initialization proccess : %.4f second(s)\n", duration(timer, 1));
    Utils::auto_printf("  input layer proccess : %.4f second(s)\n", duration(timer, 2));
    Utils::auto_printf("  tower layers proccess: %.4f second(s)\n", duration(timer, 3));
    Utils::auto_printf("  output layers proccess: %.4f second(s)\n", duration(timer, 4));
    Utils::auto_printf("Channels / Blocks :  %d / %d\n", nn_weight->residual_channels, nn_weight->residual_blocks);
    Utils::auto_printf("Tower Struct :\n");
    for (auto i = 0; i < nn_weight->residual_blocks; ++i) {
        Utils::auto_printf("  block %2d : ", i+1);
        if (nn_weight->residual_tower[i].apply_se) {
            Utils::auto_printf("ResidualBlock-SE\n");
        } else {
            Utils::auto_printf("ResidualBlock\n");
        }
    }
    Utils::auto_printf("Policy Channels : %d\n", nn_weight->policy_extract_channels);
    Utils::auto_printf("Value Channels : %d\n", nn_weight->value_extract_channels);
}

std::vector<float> get_weights_from_file(std::istream &weights_file) {
    auto weights = std::vector<float>{};
    auto line = std::string{};
#ifdef USE_FAST_PARSER
    auto w_str = std::string{};
#endif
    if (std::getline(weights_file, line)) {
        // On MacOS, if the numeric is too small, stringstream
        // can not parse the number to float, but double is ok.
        double weight;
        std::stringstream line_buffer(line);

#ifdef USE_FAST_PARSER
        while(line_buffer >> w_str) {
            bool is_ok = fast_double_parser::parse_number(w_str.c_str(), &weight);
            if (!is_ok) {
                throw "There is non-numeric in parameters";
            }
            weights.emplace_back(weight);
        }
#else 
        while(line_buffer >> weight) {
            weights.emplace_back(weight);
        }
#endif
    }
    return weights;
}

void Model::fill_fullyconnect_layer(Desc::LinearLayer &layer,
                                    std::istream &weights_file,
                                    const int in_size,
                                    const int out_size) {
    auto weights = std::vector<float>{};
    
    weights = get_weights_from_file(weights_file);
    layer.load_weights(weights);

    weights = get_weights_from_file(weights_file);
    layer.load_biases(weights);
    
    layer.load_size(in_size, out_size);
}

void Model::fill_batchnorm_layer(Desc::BatchNormLayer &layer,
                                 std::istream &weights_file,
                                 const int channels) {
    auto weights = std::vector<float>{};
    
    weights = get_weights_from_file(weights_file);
    layer.load_means(weights);

    weights = get_weights_from_file(weights_file);
    layer.load_stddevs(weights);
    
    layer.load_size(channels);
}

void Model::fill_convolution_layer(Desc::ConvLayer &layer,
                                   std::istream &weights_file,
                                   const int in_channels,
                                   const int out_channels,
                                   const int kernel_size) {
    auto weights = std::vector<float>{};
    
    weights = get_weights_from_file(weights_file);
    layer.load_weights(weights);
    
    weights = get_weights_from_file(weights_file);
    layer.load_biases(weights);
    
    layer.load_size(in_channels, out_channels, kernel_size);
}
