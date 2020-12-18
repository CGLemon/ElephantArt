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
    static Move maps2move(const int idx);
    static int move2maps(const Move &move);

    static std::string get_mapstring();

private:
    static std::unordered_map<std::uint16_t, int> moves_map;
    static std::array<Move, POLICYMAP * Board::INTERSECTIONS> policymaps_moves;
    static std::array<bool, POLICYMAP * Board::INTERSECTIONS> policymaps_valid;
};

#endif
