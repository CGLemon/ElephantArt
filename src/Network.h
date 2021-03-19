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

#ifndef NETWORK_H_INCLUDE
#define NETWORK_H_INCLUDE

#include <cassert>

#include "Model.h"
#include "Board.h"
#include "Cache.h"
#include "Position.h"

class Network {
public:
    enum Ensemble {
        DIRECT, RANDOM_SYMMETRY, SYMMETRY
    };

    ~Network();

    using Netresult = NNResult;
    using PolicyMapsPair = std::pair<float, int>;

    void initialize(const std::string &weightsfile);

    void reload_weights(const std::string &weightsfile);

    Netresult get_output(const Position *const position,
                         const Ensemble ensemble = Ensemble::DIRECT,
                         const bool read_cache = true,
                         const bool write_cache = true);

    void release_nn();

    void clear_cache();
    void set_cache_memory(const int MiB);

private:
    static constexpr auto INTERSECTIONS = Board::INTERSECTIONS;

    bool probe_cache(const Position *const position,
                     Network::Netresult &result,
                     const bool symmetry);

    Netresult get_output_internal(const Position *const position, const bool symmetry);

    Cache<Netresult> m_cache;

    std::unique_ptr<Model::NNPipe> m_forward;
    std::shared_ptr<Model::NNWeights> m_weights;

};
#endif
