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

#include "Repetition.h"
#include "ForcedCheckmate.h"

#include <cassert>

Repetition::Repetition(Position &position) : m_position(position) {}

Repetition::Result Repetition::judge() {
    const auto repetitions = m_position.get_repetitions();
    const auto cycle_length = m_position.get_cycle_length();

    if (repetitions < 2) {
        return NONE;
    }

    auto &history = m_position.get_history();
    const auto size = history.size();
    const auto last_move = m_position.get_last_move();
    const auto to_move = m_position.get_to_move();
    const auto opp_color = Board::swap_color(to_move);

    int my_ckecking_cnt = 0;
    int opp_ckecking_cnt = 0;

    assert(history[size - 2]->get_repetitions() == 1);

    if (m_position.is_check(opp_color)) {
        ++my_ckecking_cnt;
        for (int i = 1; i < cycle_length; ++i) {
            auto &board = history[history.size() - i - 1];
            if (board->is_check(Board::swap_color(board->get_to_move()))) {
                to_move == board->get_to_move() ? ++my_ckecking_cnt : ++opp_ckecking_cnt;
            }
        }

        if (my_ckecking_cnt == cycle_length/2) {
            if (my_ckecking_cnt == opp_ckecking_cnt) {
                return DRAW;
            }
            // This is perpetual check case. We lose the game.
            return LOSE;
        }
    }
    /*
    auto forced = ForcedCheckmate(m_position);
    auto ch_move = forced.find_checkmate();

    int forced_cnt = 0;
    if (ch_move.valid()) {
        auto pos_fork = std::make_shared<Position>(m_position);
        ++forced_cnt;
        for (int i = 1; i < cycle_length; i+=2) {
            pos_fork->undo_move(2);
            auto &board = history[history.size() - i - 1];
            assert(to_move == board->get_to_move());

            auto pforced = ForcedCheckmate(*pos_fork);
            if (pforced.find_checkmate().valid()) {
                ++forced_cnt;
            }
        }
        if (forced_cnt == cycle_length%2) {
            return DRAW;
        }
    }
    */

    // Perpetual pursuit

    return UNKNOWN;
}
