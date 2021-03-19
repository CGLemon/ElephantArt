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

#ifndef DECODER_H_INCLUDE
#define DECODER_H_INCLUDE

#include <array>
#include <string>
#include <unordered_map>

#include "Model.h"
#include "Types.h"
#include "BitBoard.h"
#include "Board.h"

class Decoder {
public:
    static void initialize();
    static Move maps2move(const int maps);
    static bool maps_valid(const int maps);
    static int move2maps(const Move &move);

    static int get_symmetry_maps(const int maps);

    static std::string get_mapstring();

private:
    static std::unordered_map<std::uint16_t, int> moves_map;
    static std::array<Move, POLICYMAP * Board::INTERSECTIONS> policymaps_moves;
    static std::array<bool, POLICYMAP * Board::INTERSECTIONS> policymaps_valid;
    static std::array<int, POLICYMAP * Board::INTERSECTIONS> symmetry_maps;
};

#endif
