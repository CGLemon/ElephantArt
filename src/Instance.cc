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

#include "Instance.h"

#include <cassert>

Instance::Instance(Position &position) : m_position(position) {}

Instance::Result Instance::judge() {
    const auto repetitions = m_position.get_repetitions();
    const auto cycle_length = m_position.get_cycle_length();

    if (repetitions < 2) {
        return NONE;
    }

    assert(repetitions != 3);

    auto &history = m_position.get_history();
    const auto to_move = m_position.get_to_move();

    if (m_position.is_check(to_move)) {
        auto perpetual_check = true;
        for (int i = 0; i < cycle_length; i+=2) {
            auto &board = history[history.size() - 0 - 1];
            assert(to_move == board->get_to_move());

            if (!board->is_check(to_move)) {
                perpetual_check = false;
                break;
            }
        }
        if (perpetual_check) {
            // This is perpetual check case. We lose the game.
            return LOSE;
        }
    }


    // Perpetual pursuit


    return UNKNOWN;
}
