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

#include "Evaluate.h"
#include "Position.h"

constexpr int Evaluate::PIECE_VALUE[14];
constexpr int Evaluate::POS_VALUE[14][100];


Evaluate &Evaluate::get() {
    static Evaluate eval;
    return eval;
}

int Evaluate::calc_value(Position &pos, Types::Color color) const {
    auto stable_values = pos.get_stable_values();
    auto value = 0;
    if (color == Types::RED) {
        value = stable_values[Types::RED] - stable_values[Types::BLACK];
    } else {
        value = stable_values[Types::BLACK] - stable_values[Types::RED];
    }

    return value;
}

