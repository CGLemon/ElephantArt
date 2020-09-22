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

#include <algorithm>
#include <cassert>

#include "Random.h"
#include "Zobrist.h"
#include "config.h"

constexpr std::uint64_t Zobrist::zobrist_seed;
constexpr std::uint64_t Zobrist::zobrist_empty;
constexpr std::uint64_t Zobrist::zobrist_redtomove;

std::array<std::array<std::uint64_t, Zobrist::ZOBRIST_SIZE>, 18> Zobrist::zobrist;

template<typename T>
bool is_same(std::vector<T> &array, T element) {
    auto begin = std::begin(array);
    auto end = std::end(array);
    auto res = std::find(begin, end, element);
    return res != end;
} 


void Zobrist::init_zobrist() {

    Random<random_t::XoroShiro128Plus> rng(zobrist_seed);

    auto buf = std::vector<std::uint64_t>{};

    for (int i = 0; i < 18; i++) {
        for (int j = 0; j < ZOBRIST_SIZE; j++) {
            zobrist[i][j] = rng.randuint64();
            buf.emplace_back(zobrist[i][j]);
        }
    }

    auto success = bool{true};
    for (auto &element : buf) {
        success |= (!is_same<std::uint64_t>(buf, element));
    }

    if (!success) {
        printf("The Zobrist seed is bad. Please reset a new zobrist seed.\n");
    }
}
