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

#include <memory>
#include <cassert>
#include <algorithm>

#include "ForcedCheckmate.h"
#include "Board.h"
#include "Random.h"

ForcedCheckmate::ForcedCheckmate(Position &position) : m_rootpos(position) {
    set_maxdepth(20);
    m_color = m_rootpos.get_to_move();
}

void ForcedCheckmate::set_maxdepth(int maxdepth) {
    m_maxdepth = maxdepth < BASIC_DEPTH ? BASIC_DEPTH : maxdepth;
}

Move ForcedCheckmate::find_checkmate() {
    auto movelist = m_rootpos.get_movelist();
    return find_checkmate(movelist);
}

Move ForcedCheckmate::find_checkmate(std::vector<Move> movelist) {
    auto hashbuf = std::vector<std::uint64_t>(m_maxdepth+1);
    const auto kings = m_rootpos.get_kings();

    if (m_rootpos.get_rule50_ply_left() == 0 ||
            kings[m_color] == Types::NO_VERTEX ||
            kings[Board::swap_color(m_color)] == Types::NO_VERTEX) {
        // The Game is over.
        return Move{};
    }

    shuffle(movelist);

    hashbuf[0] = m_rootpos.get_hash();

    for (const auto &move: movelist) {
        if (move.get_to() == kings[Board::swap_color(m_color)]) {
            return move;
        }

        if (m_rootpos.is_check(m_color)) {
            // We already win the game. Just find out what move capture
            // opponent king.
            continue;
        }

        auto nextpos = std::make_shared<Position>(m_rootpos);
        nextpos->do_move_assume_legal(move);
        if (nextpos->is_check(m_color)) {
            if (nextpos->get_repetitions() >= 2) {
                // This may cause the perpetual check. We may lose the game.
                // Or the other best result is draw. We don't want these results.
                continue;
            }

            const auto success = !uncheckmate_search(*nextpos, hashbuf, 1);

            if (success) {
                return move;
            }
        }
    }

    return Move{};
}

bool ForcedCheckmate::checkmate_search(Position &currpos,
                                       std::vector<std::uint64_t> &buf, int depth) const {
    if (currpos.get_rule50_ply_left() == 0 || depth > m_maxdepth) {
        return false;
    }
    const auto movelist = get_shuffle_movelist(currpos);
    const auto to_move = currpos.get_to_move();
    const auto kings = currpos.get_kings();

    buf[depth] = currpos.get_hash();

    for (const auto &move: movelist) {
        if (move.get_to() == kings[Board::swap_color(to_move)]) {
            // We capture opponent king.
            return true;
        }

        if (currpos.is_check(to_move)) {
            // We already win the game. Just find out what move capture
            // opponent king.
            continue;
        }

        auto nextpos = std::make_shared<Position>(currpos);
        nextpos->do_move_assume_legal(move);

        if (nextpos->get_repetitions() >= 2) {
            continue;
        }

        auto hash = nextpos->get_hash();
        if (std::find(std::begin(buf), std::begin(buf) + depth, hash) != std::begin(buf) + depth) {
            // This position is exist before. It may cause the perpetual check. We
            // should avoid to search it.
            continue;
        }

        if (!nextpos->is_check(to_move)) {
            // It is not forced check move.
            continue;
        }

        if (!uncheckmate_search(*nextpos, buf, depth+1)) {
            return true;
        }
    }
    // We don't find a checkmate move.
    return false;
}

bool ForcedCheckmate::uncheckmate_search(Position &currpos,
                                         std::vector<std::uint64_t> &buf, int depth) const {
    if (currpos.get_rule50_ply_left() == 0 || depth > m_maxdepth) {
        return true;
    }
    const auto movelist = get_shuffle_movelist(currpos);
    const auto to_move = currpos.get_to_move();
    const auto kings = currpos.get_kings();
    
    buf[depth] = currpos.get_hash();

    for (const auto &move: movelist) {
        if (move.get_to() == kings[Board::swap_color(to_move)]) {
            // We capture opponent king.
            return true;
        }

        if (currpos.is_check(to_move)) {
            // We already win the game. Just find out what move capture
            // opponent king.
            continue;
        }

        auto nextpos = std::make_shared<Position>(currpos);
        nextpos->do_move_assume_legal(move);

        if (nextpos->is_check(Board::swap_color(to_move))) {
            // We are already lose.
            continue;
        }

        auto hash = nextpos->get_hash();
        if (std::find(std::begin(buf), std::begin(buf) + depth, hash) != std::begin(buf) + depth) {
            // This position is exist before. We don't need to waste time
            // to search it.
            return true;
        }

        if (!checkmate_search(*nextpos, buf, depth+1)) {
            return true;
        }
    }

    // We don't find a uncheckmate move.
    return false;
}

void ForcedCheckmate::shuffle(std::vector<Move> &movelist) const {
    auto rng = Random<random_t::SplitMix_64>::get_Rng();
    std::shuffle(std::begin(movelist), std::end(movelist), rng);
}

std::vector<Move> ForcedCheckmate::get_shuffle_movelist(Position &pos) const {
    auto movelist = pos.get_movelist();
    shuffle(movelist);
    return movelist;
}
