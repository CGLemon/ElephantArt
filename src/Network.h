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

#ifndef NETWORK_H_INCLUDE
#define NETWORK_H_INCLUDE

#include <cassert>

#include "Model.h"
#include "Board.h"
#include "Cache.h"
#include "Position.h"

class Network {
public:
    ~Network();

    enum Ensemble { NONE, DIRECT, RANDOM_SYMMETRY /* , AVERAGE */ };

    using Netresult = NNResult;
    using PolicyVertexPair = std::pair<float, int>;

    void initialize(const int playouts, const std::string &weightsfile);

    void reload_weights(const std::string &weightsfile);

    Netresult get_output(const Position *const position,
                         const Ensemble ensemble,
                         const int symmetry = -1,
                         const bool read_cache = true,
                         const bool write_cache = true);

    void clear_cache();

    void release_nn();

    void set_playouts(const int playouts);


private:
    static constexpr auto INTERSECTIONS = Board::INTERSECTIONS;
    static constexpr auto NUM_SYMMETRIES = Board::NUM_SYMMETRIES;
    static constexpr auto IDENTITY_SYMMETRY = Board::IDENTITY_SYMMETRY;

    bool probe_cache(const Position *const position,
                     Network::Netresult &result,
                     const int symmetry = -1);

    Netresult get_output_internal(const Position *const position,
                                  const int symmetry);
  
    Netresult get_output_form_cache(const Position *const position);

    Cache<Netresult> m_cache;

    std::unique_ptr<Model::NNpipe> m_forward;
    std::shared_ptr<Model::NNweights> m_weights;

};
#endif
