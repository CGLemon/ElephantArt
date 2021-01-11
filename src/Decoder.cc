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

#include "Decoder.h"
#include "Model.h"

#include <cassert>
#include <sstream>

std::array<Move, POLICYMAP * Board::INTERSECTIONS> Decoder::policymaps_moves;
std::array<bool, POLICYMAP * Board::INTERSECTIONS> Decoder::policymaps_valid;
std::unordered_map<std::uint16_t, int> Decoder::moves_map;

void Decoder::initialize() {
    const auto in_boundary = [&](const int x, const int y) -> bool {
        if (x < 0 || x >= Board::WIDTH) {
            return false;
        }
        if (y < 0 || y >= Board::HEIGHT) {
            return false;
        }
        return true;
    };

    const auto fill_maps = [&](const int m_offset,
                               const Types::Vertices from_vtx,
                               const int to_x,
                               const int to_y) -> void {
        auto& move = policymaps_moves[m_offset];
        if (in_boundary(to_x, to_y)) {
            const auto to_vtx = static_cast<Types::Vertices>(Board::get_vertex(to_x, to_y));
            move = Move(from_vtx, to_vtx);
            assert(move.is_ok());
        } else {
            policymaps_valid[m_offset] = false;
            assert(!move.valid());
        }
    };

    std::fill(std::begin(policymaps_moves), std::end(policymaps_moves), Move{});
    std::fill(std::begin(policymaps_valid), std::end(policymaps_valid), true);
 
    // planes |  1 - 18 | file moves
    for (int p = 0; p < 18; ++p) {
        for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
            const auto x = Board::get_x(v);
            const auto y = Board::get_y(v);
            
            if (!in_boundary(x,y)) {
                continue;
            }
            
            const auto idx = Board::get_index(x, y);
            const auto m_offset = idx + p * Board::INTERSECTIONS;

            const int ms[18] = {
                1, 2, 3, 4, 5, 6, 7, 8, 9,
                -1, -2, -3, -4, -5, -6, -7, -8, -9
            };
            const auto to_x = x;
            const auto to_y = y + ms[p];
            fill_maps(m_offset, v, to_x, to_y);
        }
    }

    // planes | 19 - 34 | rank moves
    for (int p = 0; p < 16; ++p) {
        for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
            const auto x = Board::get_x(v);
            const auto y = Board::get_y(v);
            
            if (!in_boundary(x,y)) {
                continue;
            }
            const auto idx = Board::get_index(x, y);
            const auto m_offset = idx + (p + 18) * Board::INTERSECTIONS;

            const int ms[16] = {
                1, 2, 3, 4, 5, 6, 7, 8,
                -1, -2, -3, -4, -5, -6, -7, -8
            };
            const auto to_x = x + ms[p];
            const auto to_y = y;
            fill_maps(m_offset, v, to_x, to_y);
        }
    }
    
    // planes | 35 - 42 | horse moves
    for (int p = 0; p < 8; ++p) {
        for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
            const auto x = Board::get_x(v);
            const auto y = Board::get_y(v);
            
            if (!in_boundary(x,y)) {
                continue;
            }
            const auto idx = Board::get_index(x, y);
            const auto m_offset = idx + (p + 34) * Board::INTERSECTIONS;

            const int ms[8][2] = {
                {2,1}, {2,-1}, {-2,1}, {-2,-1},
                {1,2}, {1,-2}, {-1,2}, {-1,-2}
            };
            const auto to_x = x + ms[p][0];
            const auto to_y = y + ms[p][1];
            fill_maps(m_offset, v, to_x, to_y);
        }
    }
    
    // planes | 43 - 46 | advisor moves
    for (int p = 0; p < 4; ++p) {
        for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
            const auto x = Board::get_x(v);
            const auto y = Board::get_y(v);
            
            if (!in_boundary(x,y)) {
                continue;
            }
            const auto idx = Board::get_index(x, y);
            const auto m_offset = idx + (p + 42) * Board::INTERSECTIONS;

            const int ms[4][2] = {
                {1,1}, {1,-1}, {-1,1}, {-1,-1}
            };
            const auto to_x = x + ms[p][0];
            const auto to_y = y + ms[p][1];
            fill_maps(m_offset, v, to_x, to_y);
        }
    }
    
    // planes | 47 - 50 | elephant moves
    for (int p = 0; p < 4; ++p) {
        for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
            const auto x = Board::get_x(v);
            const auto y = Board::get_y(v);
            
            if (!in_boundary(x,y)) {
                continue;
            }
            const auto idx = Board::get_index(x, y);
            const auto m_offset = idx + (p + 46) * Board::INTERSECTIONS;

            const int ms[4][2] = {
                {2,2}, {2,-2}, {-2,2}, {-2,-2}
            };
            const auto to_x = x + ms[p][0];
            const auto to_y = y + ms[p][1];
            fill_maps(m_offset, v, to_x, to_y);
        }
    }

    for (int idx = 0; idx < POLICYMAP * Board::INTERSECTIONS; ++idx) {
        const auto &move = policymaps_moves[idx];
        if (move.valid()) {
            assert(moves_map.find(move.get_data()) == std::end(moves_map));
            moves_map.emplace(move.get_data(), idx);
        }
    }
}

Move Decoder::maps2move(const int idx) {
    assert(idx >= 0 && idx < POLICYMAP * Board::INTERSECTIONS);
    return policymaps_moves[idx];
}

bool Decoder::maps_valid(const int idx) {
    return policymaps_valid[idx];
}

int Decoder::move2maps(const Move &move) {
    const auto iter = moves_map.find(move.get_data());
    assert(iter != std::end(moves_map));
    const auto maps = iter->second;
    return maps;
}

std::string Decoder::get_mapstring() {
    auto out = std::ostringstream{}; 
    for (int p = 0; p < POLICYMAP; ++p) {
        out << "maps : " << p+1 << std::endl;
        for (int y = 0; y < Board::HEIGHT; ++y) {
            for (int x = 0; x < Board::WIDTH; ++x) {
                const auto idx = Board::get_index(x, y);
                out << policymaps_moves[p * Board::INTERSECTIONS + idx].to_string() << " ";
            }
            out << std::endl;
        }
        out << std::endl;
    }
    return out.str();
}
