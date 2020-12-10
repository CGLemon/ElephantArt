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

#include "Position.h"

#include <iterator>
#include <sstream>


void Position::init_game() {
    m_startboard = 0;
    m_history.clear();
    board.reset_board();
    push_board();
}

void Position::display() const {
    board.dump_board();
}

void Position::push_board() {
    m_history.emplace_back(std::make_shared<const Board>(board));
    assert(get_movenum() == (int)m_history.size());
}

bool Position::fen2board(std::string &fen) {
    return board.fen2board(fen);
}

bool Position::is_legal(Move move) const {
    return board.is_legal(move);
}

void Position::do_move_assume_legal(Move move) {
    board.do_move(move);
    push_board();
}

bool Position::do_move(Move move) {
    if (!is_legal(move)) {
        return false;
    }

    do_move_assume_legal(move);

    return true;
}

bool Position::do_textmove(std::string smove) {
    const auto move = board.text2move(smove);
    if (move.valid()) {
        return do_move(move);
    }
    return false;
}

std::vector<Move> Position::get_movelist() const {

    auto movelist = std::vector<Move>{};
    const auto color = get_to_move();
    board.generate_movelist(color, movelist);

    return movelist;
}

bool Position::undo() {

    const auto movenum = get_movenum();
    if (movenum == 1) {
        return false;
    }

    m_history.resize(movenum - 1);
    board = *m_history[movenum - 2];

    assert(get_movenum() == movenum - 1);
    assert(get_movenum() == (int)m_history.size());

    return true;
}

bool Position::position(std::string &fen, std::string& moves) {

    // first : Set the position.
    auto fork_board = std::make_shared<Board>(board);
    auto success = fork_board->fen2board(fen);

    if (!success) {
        return false;
    }

    // second : Check out the position is exist.
    auto current_movenum = fork_board->get_movenum();
    assert(current_movenum >= 1);
    if (!(fork_board->get_hash() == m_history[current_movenum-1]->get_hash())) {
        return false;
    } 

    if (moves.empty()) {
        m_startboard = current_movenum-1;
        m_history.resize(current_movenum);
        board = *m_history[current_movenum-1];
        return true;
    }


    // third : Do moves.
    auto chain_board = std::vector<std::shared_ptr<Board>>{};
    bool moves_success = true;
    auto move_cnt = size_t{0};

    auto moves_stream = std::stringstream{moves};
    auto move_str = std::string{};

    while (moves_stream >> move_str) {
        const auto move = fork_board->text2move(move_str);
        ++move_cnt;
        if (move.valid()) {
            if (fork_board->is_legal(move)) {
                fork_board->do_move(move);
                chain_board.emplace_back(std::make_shared<Board>(*fork_board));
            }
        }

        if (move_cnt != chain_board.size()) {
            moves_success = false;
        }
    }

    if (moves_success) {
        m_history.resize(current_movenum);
        for (auto b : chain_board) {
            m_history.emplace_back(b);
        }
        current_movenum += move_cnt;
        m_startboard = current_movenum-1;
        board = *m_history[current_movenum-1];
        assert(get_movenum() == current_movenum - 1);
        assert(get_movenum() == (int)m_history.size());
    }

    return moves_success;
}

Types::Color Position::get_to_move() const {
    return board.get_to_move();
}

int Position::get_movenum() const {
    return board.get_movenum();
}

std::uint64_t Position::get_hash() const {
    return board.get_hash();
}

Move Position::get_last_move() const {
    return board.get_last_move();
}

const std::shared_ptr<const Board> Position::get_past_board(const int p) const {
    const auto movenum = get_movenum();
    assert(0 <= p && p < movenum);
    return m_history[movenum - p - 1];
}
