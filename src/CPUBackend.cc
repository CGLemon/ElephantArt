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

#include "CPUBackend.h"
#include "Board.h"
#include "Utils.h"
#include "Model.h"

void CPUBackend::initialize(std::shared_ptr<Model::NNWeights> weights) {
    reload(weights);
}

void CPUBackend::reload(std::shared_ptr<Model::NNWeights> weights) {
    if (m_weights != nullptr) {
        m_weights.reset();
    }
    m_weights = weights;
}

void CPUBackend::forward(const std::vector<float> &planes,
                         const std::vector<float> &features,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_val) {

    using Convolve3 = Convolve<3>;

    const auto output_channels = m_weights->residual_channels;
    auto max_channels = std::max({INPUT_CHANNELS,
                                  output_channels,
                                  m_weights->policy_extract_channels,
                                  m_weights->policy_map});
    
    auto workspace_size = Convolve3::get_workspace_size(max_channels);
    auto workspace = std::vector<float>(workspace_size);

    auto conv_out = std::vector<float>(output_channels * Board::NUM_INTERSECTIONS);
    auto conv_in = std::vector<float>(output_channels * Board::NUM_INTERSECTIONS);
    auto res = std::vector<float>(output_channels * Board::NUM_INTERSECTIONS);
    
    // input
    Convolve3::Forward(INPUT_CHANNELS, output_channels,
                       planes,
                       m_weights->input_conv.weights,
                       workspace, conv_out);

    Batchnorm::Forward(output_channels, conv_out,
                       m_weights->input_bn.means,
                       m_weights->input_bn.stddevs,
                       nullptr, false);

    InputPool::Forward(INPUT_FEATURES, 2 * output_channels, output_channels,
                       features,
                       m_weights->input_fc1.weights,
                       m_weights->input_fc1.biases,
                       m_weights->input_fc2.weights,
                       m_weights->input_fc2.biases,
                       conv_out);

    // residual tower
    const auto residuals =  m_weights->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        const auto tower_channels = m_weights->residual_channels;
        const auto tower_ptr = m_weights->residual_tower.data() + i;

        std::swap(conv_in, conv_out);
        
        Convolve3::Forward(tower_channels, tower_channels,
                           conv_in,
                           tower_ptr->conv_1.weights,
                           workspace, conv_out);

        Batchnorm::Forward(tower_channels, conv_out,
                           tower_ptr->bn_1.means,
                           tower_ptr->bn_1.stddevs);

        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);
        Convolve3::Forward(tower_channels, tower_channels,
                           conv_in,
                           tower_ptr->conv_2.weights,
                           workspace, conv_out);

        if (tower_ptr->apply_se) {
            Batchnorm::Forward(tower_channels, conv_out,
                               tower_ptr->bn_2.means,
                               tower_ptr->bn_2.stddevs,
                               nullptr, false);
       
            const size_t se_size = tower_ptr->se_size;
            SEUnit::Forward(tower_channels, se_size,
                            conv_out, res,
                            tower_ptr->extend.weights,
                            tower_ptr->extend.biases,
                            tower_ptr->squeeze.weights,
                            tower_ptr->squeeze.biases);
        
        } else {
             Batchnorm::Forward(tower_channels, conv_out,
                                tower_ptr->bn_2.means,
                                tower_ptr->bn_2.stddevs,
                                res.data());
        }
    }
    
    // policy head

    const auto policy_extract_channels = m_weights->policy_extract_channels;
    auto policy_conv = std::vector<float>(policy_extract_channels * Board::NUM_INTERSECTIONS);

    Convolve3::Forward(output_channels, policy_extract_channels,
                       conv_out,
                       m_weights->p_ex_conv.weights,
                       workspace, policy_conv);
    
    Batchnorm::Forward(policy_extract_channels, policy_conv,
                       m_weights->p_ex_bn.means,
                       m_weights->p_ex_bn.stddevs);
    
    Convolve3::Forward(policy_extract_channels, POLICYMAP,
                       policy_conv,
                       m_weights->p_map.weights,
                       workspace, output_pol);

    AddSpatialBias::Forward(POLICYMAP, output_pol, m_weights->p_map.biases);

    // value head
    const auto value_extract_channels = m_weights->value_extract_channels;
    auto value_conv = std::vector<float>(value_extract_channels * Board::NUM_INTERSECTIONS);
    auto value_fc = std::vector<float>(VALUELAYER);
    
    Convolve1::Forward(output_channels, value_extract_channels,
                       conv_out,
                       m_weights->v_ex_conv.weights,
                       value_conv);
    
    Batchnorm::Forward(value_extract_channels, value_conv,
                       m_weights->v_ex_bn.means,
                       m_weights->v_ex_bn.stddevs);
    
    FullyConnect::Forward(value_extract_channels * Board::NUM_INTERSECTIONS, VALUELAYER,
                          value_conv,
                          m_weights->v_fc1.weights,
                          m_weights->v_fc1.biases,
                          value_fc, true);

    FullyConnect::Forward(VALUELAYER, VLAUEMISC_LAYER,
                          value_fc,
                          m_weights->v_fc2.weights,
                          m_weights->v_fc2.biases,
                          output_val, false);

}

void CPUBackend::destroy() {
    // Do nothing.
}

void CPUBackend::release() {
    if (m_weights != nullptr) {
        m_weights.reset();
    }
    m_weights = nullptr;
}

bool CPUBackend::valid() {
    return m_weights->loaded;
}
