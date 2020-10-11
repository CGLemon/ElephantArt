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
bool collision(std::vector<T> &array) {
    const auto s = array.size();
    if (s == 1) {
        return false;
    }

    for (auto i = size_t{0}; i < (s-1); ++i) {
        auto begin = std::cbegin(array);
        auto element = std::next(begin, i);
        auto start = std::next(element, 1);
        auto end = std::cend(array);
        auto res = std::find(start, end, *element);
        if (res != end) {
            return true;
        }
    }
    return false;
}

void Zobrist::init_zobrist() {

    Random<random_t::XoroShiro128Plus> rng(zobrist_seed);
    while (true) {
        auto buf = std::vector<std::uint64_t>{};

        for (int i = 0; i < 18; i++) {
            for (int j = 0; j < ZOBRIST_SIZE; j++) {
                zobrist[i][j] = rng.randuint64();
                buf.emplace_back(zobrist[i][j]);
            }
        }

        if (!collision(buf)) {
            break;
        }
    }
}
