/*
    This file is part of Saya.
    Copyright (C) 2020 Hung-Zhe, Lin

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

#ifndef ZOBRIST_H_INCLUDE
#define ZOBRIST_H_INCLUDE

#include <array>
#include <random>
#include <vector>

#include "Board.h"

class Zobrist {
private:
    static constexpr int ZOBRIST_SIZE = Board::NUM_VERTICES;

    static constexpr std::uint64_t zobrist_seed = 0xabcdabcd12345678;

public:
    static constexpr std::uint64_t zobrist_empty = 0x1234567887654321;

    static constexpr std::uint64_t zobrist_redtomove = 0xabcdabcdabcdabcd;

    // Including all type of red, black pieces and empty, invalid. 
    static std::array<std::array<std::uint64_t, ZOBRIST_SIZE>, 18> zobrist;

    static void init_zobrist();
};

#endif
