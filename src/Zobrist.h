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

#ifndef ZOBRIST_H_INCLUDE
#define ZOBRIST_H_INCLUDE

#include <array>
#include <random>
#include <vector>

#include "BitBoard.h"

class Zobrist {
private:
    using KEY = std::uint64_t;

    static constexpr auto ZOBRIST_SIZE = BITBOARD_NUM_VERTICES;
    static constexpr KEY zobrist_seed = 0xabcdabcd12345678;

public:
    static constexpr KEY zobrist_empty = 0x1234567887654321;
    static constexpr KEY zobrist_redtomove = 0xabcdabcdabcdabcd;

    static std::array<std::array<KEY, ZOBRIST_SIZE>, 18> zobrist;
    static std::array<KEY, 200> zobrist_positions;

    static void init_zobrist();
};

#endif
