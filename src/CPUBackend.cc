#include "CPUBackend.h"
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

#include "Board.h"
#include "Utils.h"

void CPUbackend::initialize(std::shared_ptr<Model::NNweights> weights) {
    reload(weights);
}

void CPUbackend::reload(std::shared_ptr<Model::NNweights> weights) {
    if (m_weights != nullptr) {
        m_weights.reset();
    }
    m_weights = weights;
}

void CPUbackend::forward(const std::vector<float> &planes,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_val) {

    using Convolve3 = Convolve<3>;
    
    
    const auto input_channels = m_weights->input_channels;
    const auto output_channels = m_weights->residual_channels;
    
    auto max_channels = std::max({input_channels,
                                  output_channels,
                                  m_weights->policy_extract_channels,
                                  m_weights->policy_map});
    
    auto workspace_size = Convolve3::get_workspace_size(max_channels);
    auto workspace = std::vector<float>(workspace_size);

    auto conv_out = std::vector<float>(output_channels * Board::INTERSECTIONS);
    auto conv_in = std::vector<float>(output_channels * Board::INTERSECTIONS);
    auto res = std::vector<float>(output_channels * Board::INTERSECTIONS);
    
    // input
    Convolve3::Forward(input_channels, output_channels,
                       planes,
                       m_weights->input_conv.weights,
                       workspace, conv_out);

    Batchnorm::Forward(output_channels, conv_out,
                       m_weights->input_bn.means,
                       m_weights->input_bn.stddevs,
                       nullptr, true);
    
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
    const auto policy_map_channels = m_weights->policy_map;
    const auto policy_channels = std::max(policy_map_channels, policy_extract_channels);
    auto policy_conv = std::vector<float>(policy_channels * Board::INTERSECTIONS);

    Convolve3::Forward(output_channels, policy_extract_channels,
                       conv_out,
                       m_weights->p_ex_conv.weights,
                       workspace, policy_conv);
    
    Batchnorm::Forward(policy_extract_channels, policy_conv,
                       m_weights->p_ex_bn.means,
                       m_weights->p_ex_bn.stddevs);
    
    Convolve3::Forward(policy_extract_channels, policy_map_channels,
                       policy_conv,
                       m_weights->p_map.weights,
                       workspace, output_pol);
    
    AddSpatialBias::Forward(policy_map_channels, output_pol, m_weights->p_map.biases);
    
    // value head
    const auto value_extract_channels = m_weights->value_extract_channels;
    auto value_conv = std::vector<float>(policy_channels * Board::INTERSECTIONS);
    auto value_fc = std::vector<float>(256);
    
    Convolve1::Forward(output_channels, value_extract_channels,
                       conv_out,
                       m_weights->v_ex_conv.weights,
                       value_conv);
    
    Batchnorm::Forward(value_extract_channels, value_conv,
                       m_weights->v_ex_bn.means,
                       m_weights->v_ex_bn.stddevs);
    
    FullyConnect::Forward(value_extract_channels * Board::INTERSECTIONS, 256,
                          value_conv,
                          m_weights->v_fc1.weights,
                          m_weights->v_fc1.biases,
                          value_fc, true);
    
    FullyConnect::Forward(256, 3,
                          value_fc,
                          m_weights->v_fc2.weights,
                          m_weights->v_fc2.biases,
                          output_val, false);

}

void CPUbackend::destroy() {
    
}

void CPUbackend::release() {
    if (m_weights != nullptr) {
        m_weights.reset();
    }
    m_weights = nullptr;
}

bool CPUbackend::valid() {
    return m_weights->loaded;
}
