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

#include "Network.h"

#ifdef USE_ZLIB
#include "zlib.h"
#endif

#include "CPUBackend.h"
#include "Board.h"
#include "Position.h"
#include "Random.h"
#include "Utils.h"
#include "Blas.h"
#include "config.h"

#ifdef USE_CUDA
#include "CUDABackend.h"
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_OPENBLAS
#ifndef __APPLE__
#include <cblas.h>
#endif
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

Network::~Network() {
    m_forward->destroy();  
}

void Network::initialize(const std::string &weightsfile) {

#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    UCCI_DEBUG << "BLAS Core:" << ' ' << openblas_get_corename() << std::endl;
#endif
#ifdef USE_MKL
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    UCCI_DEBUG << "BLAS core: MKL" << ' ' << Version.Processor << std::endl;
#endif
#endif

#ifdef USE_EIGEN
    UCCI_DEBUG << "BLAS Core: Eigen" << ' '
                   << EIGEN_WORLD_VERSION << '.' << EIGEN_MAJOR_VERSION << '.' << EIGEN_MINOR_VERSION << ' '
                   << "library." << std::endl;
#endif

    if (option<int>("cache_playouts") != 0) {
        m_cache.set_playouts(option<int>("cache_playouts"));
    } else {
        m_cache.set_memory(option<int>("cache_size"));
    }

#ifdef USE_CUDA
    using backend = CUDABackend;
#else
    using backend = CPUBackend;
#endif

    m_forward = std::make_unique<backend>();

    m_weights = std::make_shared<Model::NNWeights>();
    Model::load_weights(weightsfile, m_weights);

    m_forward->initialize(m_weights);

    if (m_weights->loaded) {
        // Do nothing...
    }

    m_weights.reset();
    m_weights = nullptr;
}

void Network::reload_weights(const std::string &weightsfile) {
    if (m_weights != nullptr) {
        return;
    }
    m_weights = std::make_shared<Model::NNWeights>();
    Model::load_weights(weightsfile, m_weights);

    m_forward->reload(m_weights);

    if (m_weights->loaded) {
        // Do nothing...
    }
    m_weights.reset();
    m_weights = nullptr;
}

void Network::set_cache_memory(const int MiB) {
    m_cache.set_memory(MiB);
}

bool Network::probe_cache(const Position *const position,
                          Network::Netresult &result) {
    // TODO: Cache the all symmetry position in early game.

    if (m_cache.lookup(position->get_hash(), result)) {
        return true;
    }
    return false;
}

void dummy_forward(std::vector<float> &policy,
                   std::vector<float> &value) {

    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    auto dist = std::uniform_real_distribution<float>(0.0, 1.0);

    for (auto &p : policy) {
        p = dist(rng);
    }
    const auto p_acc = std::accumulate(std::begin(policy),
                                       std::end(policy), 0.0f);
    for (auto &p : policy) {
        p /= p_acc;
    }

    for (auto &v : value) {
        v = dist(rng);
    }
    const auto v_acc = std::accumulate(std::begin(value),
                                       std::begin(value)+3, 0.0f);

    for (int idx = 0; idx < 3; ++idx) {
        value[idx] = value[idx] / v_acc;
    }
}

Network::Netresult
Network::get_output_internal(const Position *const position, const bool symmetry) {
    auto policy_out = std::vector<float>(POLICYMAP * INTERSECTIONS);
    auto value_out = std::vector<float>(VLAUEMISC_LAYER);

    auto input_planes = Model::gather_planes(position, symmetry);
    auto input_features = Model::gather_features(position);

    if (m_forward->valid()) {
        m_forward->forward(input_planes, input_features, policy_out, value_out);
    } else {
        // If we didn't load the network yet, output the random result.
        dummy_forward(policy_out, value_out);
    }

    // TODO: Remove "softmax_pol_temp" and "softmax_wdl_temp" to UCCI Option.
    const auto result = Model::get_result(policy_out,
                                          value_out,
                                          option<float>("softmax_pol_temp"),
                                          option<float>("softmax_wdl_temp"),
                                          symmetry);

    return result;
}

Network::Netresult
Network::get_output(const Position *const position,
                    const Network::Ensemble ensemble,
                    const bool read_cache,
                    const bool write_cache) {
    Netresult result;
    bool symm = false;

    if (ensemble == DIRECT) {
        symm = static_cast<bool>(Board::IDENTITY_SYMMETRY);
        assert(!symm);
    } else if (ensemble == RANDOM_SYMMETRY) {
        auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
        symm = static_cast<bool>(rng.randfix<2>());
    } else if (ensemble == SYMMETRY) {
        symm = static_cast<bool>(Board::MIRROR_SYMMETRY);
        assert(symm);
    }

    if (read_cache) {
        // Get result from cache, if the it is in the cache memory.
        if (probe_cache(position, result)) {
            return result;
        }
    }

    result = get_output_internal(position, symm);

    if (write_cache) {
        // Write result to cache, if the it is not in the cache memory.
        m_cache.insert(position->get_hash(), result);
    }
    return result;
}

void Network::release_nn() {
    m_forward->release();
}

void Network::clear_cache() {
    m_cache.clear();
}
