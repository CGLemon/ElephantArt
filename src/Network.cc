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
#include <cblas.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

using namespace Utils;

Network::~Network() {
    m_forward->destroy();  
}

void Network::initialize(const int playouts, const std::string &weightsfile) {

#ifndef __APPLE__
#ifdef USE_OPENBLAS
    openblas_set_num_threads(1);
    auto_printf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
    mkl_set_num_threads(1);
    MKLVersion Version;
    mkl_get_version(&Version);
    auto_printf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif

#ifdef USE_EIGEN
    auto_printf("BLAS Core: built-in Eigen %d.%d.%d library.\n",
                EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif
    set_playouts(playouts);

#ifdef USE_CUDA
    using backend = CUDAbackend;
#else
    using backend = CPUbackend;
#endif

    m_weights = std::make_shared<Model::NNweights>();
    Model::load_weights(weightsfile, m_weights);

    m_forward = std::make_unique<backend>();
    m_forward->initialize(m_weights);

    if (m_weights->loaded) {
        auto_printf("Weights are pushed down\n");
    }

    m_weights.reset();
    m_weights = nullptr;
}

void Network::reload_weights(const std::string &weightsfile) {
    if (m_weights != nullptr) {
        return;
    }
    m_weights = std::make_shared<Model::NNweights>();
    Model::load_weights(weightsfile, m_weights);

    m_forward->reload(m_weights);

    if (m_weights->loaded) {
        auto_printf("Weights are pushed down\n");
    }
    m_weights.reset();
    m_weights = nullptr;
}

void Network::set_playouts(const int playouts) {
    const size_t cache_size = option<int>("cache_moves") * playouts;
    m_cache.resize(cache_size);
}

bool Network::probe_cache(const Position *const position,
                          Network::Netresult &result,
                          const int symmetry) {

    bool success = false;
    if (symmetry == -1 || symmetry == IDENTITY_SYMMETRY) {
        success = m_cache.lookup(position->get_hash(), result);
    } else {
        success = m_cache.lookup(position->calc_hash(symmetry), result);
    }
    return success;
}

void dummy_forward(std::vector<float> &policy,
                   std::vector<float> &value) {

    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    auto dis = std::uniform_real_distribution<float>(0.0, 1.0);
    for (auto &p : policy) {
        p = dis(rng);
    }
    const auto acc = std::accumulate(std::begin(policy),
                                     std::end(policy), 0.0f);
    for (auto &p : policy) {
        p /= acc;
    }

    for (auto &v : value) {
        v = (1.0f/3.0f);
    }
}

Network::Netresult Network::get_output_internal(const Position *const position,
                                                const int symmetry) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    auto policy_out = std::vector<float>(POLICYMAP * INTERSECTIONS);
    auto winrate_out = std::vector<float>(WINRATELAYER);

    const auto input_planes = Model::gather_planes(position, symmetry);

    if (m_forward->valid()) {
        m_forward->forward(input_planes, policy_out, winrate_out);
    } else {
        dummy_forward(policy_out, winrate_out);
    }


    const auto result = Model::get_result(policy_out,
                                          winrate_out,
                                          option<float>("softmax_temp"),
                                          option<float>("softmax_temp"),
                                          symmetry);

    return result;
}

Network::Netresult
Network::get_output(const Position *const position,
                    const Ensemble ensemble,
                    const int symmetry,
                    const bool read_cache,
                    const bool write_cache) {

    Netresult result;

    if (read_cache && symmetry == -1) {
        if (probe_cache(position, result)) {
            return result;
        }
    }

    if (ensemble == NONE) {
        assert(symmetry == -1);
        result = get_output_internal(position, IDENTITY_SYMMETRY);
    } else if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
        result = get_output_internal(position, symmetry);
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
        const auto rand_sym = rng.randfix<NUM_SYMMETRIES>();
        result = get_output_internal(position, rand_sym);
    }

    if (write_cache && symmetry == -1) {
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
