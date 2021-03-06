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

#include "Position.h"
#include "Zobrist.h"
#include "Instance.h"

#include <queue>
#include <iterator>
#include <sstream>

void Position::init_game(const int tag) {
    set_max_moves(150);
    m_startboard = 0;
    position_hash = Zobrist::zobrist_positions[tag];
    m_history.clear();
    board.reset_board();
    push_board();
    resigned = Types::INVALID_COLOR;
}

void Position::display() const {
    const auto lastmove = get_last_move();
    if (option<bool>("using_traditional_chinese")) {
        board.dump_board<Types::TRADITIONAL_CHINESE>(lastmove);
    } else {
        board.dump_board<Types::ASCII>(lastmove);
    }
}

void Position::do_resigned() {
    assert(resigned == Types::INVALID_COLOR);
    resigned = get_to_move();
}

void Position::push_board() {
    m_history.emplace_back(std::make_shared<const Board>(board));
}

bool Position::fen(std::string &fen) {
    auto fork_board = std::make_shared<Board>(board);
    auto success = fork_board->fen2board(fen);

    if (!success) {
        return false;
    }

    m_history.clear();
    m_startboard = 0;

    board = *fork_board;
    push_board();

    return true;
}

bool Position::is_legal(Move move) {
    return board.is_legal(move);
}

void Position::do_move_assume_legal(Move move) {
    board.do_move_assume_legal(move);
    push_board();
    if (is_capture()) {
        m_startboard = m_history.size() - 1;
    }
    compute_repetitions();
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

std::vector<Move> Position::get_movelist() {
    auto movelist = std::vector<Move>{};
    const auto color = get_to_move();
    board.generate_movelist(color, movelist);

    return movelist;
}

bool Position::position(std::string &fen, std::string &moves) {
    // First: Set the Fen.
    auto fork_board = std::make_shared<Board>(board);
    auto success = fork_board->fen2board(fen);

    if (!success) {
        return false;
    }

    // Second: Scan the moves.
    auto chain_moves = std::queue<Move>{};
    bool moves_success = true;
    auto move_cnt = size_t{0};

    if (!moves.empty()) {
        auto moves_stream = std::stringstream{moves};
        auto move_str = std::string{};

        while (moves_stream >> move_str) {
            const auto move = Board::text2move(move_str);
            if (move.valid()) {
                if (fork_board->is_legal(move)) {
                    fork_board->do_move_assume_legal(move);
                    chain_moves.emplace(move);
                }
            }

            if (++move_cnt != chain_moves.size()) {
                moves_success = false;
                break;
            }
        }
    }

    if (moves_success) {
        Position::fen(fen);
        while(!chain_moves.empty()) {
            do_move_assume_legal(chain_moves.front());
            chain_moves.pop();
        }
        assert(m_history.size() == move_cnt + 1);
    }

    return moves_success;
}

bool Position::gameover(bool searching) {
    return get_winner(searching) != Types::INVALID_COLOR;
}

Types::Color Position::get_winner(bool searching) {
    if (resigned != Types::INVALID_COLOR) {
        if (resigned == Types::EMPTY_COLOR) {
            return Types::EMPTY_COLOR;
        }
        return Board::swap_color(resigned);
    }

    // According Asian Xiangqi Federation rules, if the current
    // player can eat opponent king. The current player win the
    // game.
    const auto to_move = get_to_move();
    if (!searching && is_check(to_move)) {
        return to_move;
    }

    const auto kings = get_kings();
    if (kings[Types::RED] == Types::NO_VERTEX) {
        return Types::BLACK;
    } else if (kings[Types::BLACK] == Types::NO_VERTEX) {
        return Types::RED;
    }

    if (get_movenum() > m_maxmoves) {
        return Types::EMPTY_COLOR;
    }

    auto instance = Instance(*this);
    auto res = instance.judge();
    if (res == Instance::DRAW) {
        return Types::EMPTY_COLOR;
    } else if (res == Instance::LOSE) {
        return Board::swap_color(to_move);
    } else if (res == Instance::UNKNOWN) {
        // The program has no idea what the result is. Maybe the opponent 
        // is lose, or draw. But we are so gentle. We simply think that the 
        // result is draw. No consider lose result.
        return Types::EMPTY_COLOR;
    }

    return Types::INVALID_COLOR;
}

Types::Color Position::get_to_move() const {
    return board.get_to_move();
}

int Position::get_movenum() const {
    return board.get_movenum();
}

int Position::get_gameply() const {
    return board.get_gameply();
}

std::uint64_t Position::get_hash() const {
    return board.get_hash() ^ position_hash;
}

std::uint64_t Position::calc_hash() const {
    return board.calc_hash() ^ position_hash;
}

Move Position::get_last_move() const {
    return board.get_last_move();
}

Types::Piece_t Position::get_piece_type(const Types::Vertices vtx) const {
    return board.get_piece_type(vtx);
}

Types::Piece Position::get_piece(const Types::Vertices vtx) const {
    return board.get_piece(vtx);
}

const std::shared_ptr<const Board> Position::get_past_board(const int p) const {
    const auto size = m_history.size();
    assert(0 <= p && p <= (int)size - 1);
    return m_history[size - p - 1];
}

void Position::compute_repetitions() {
    int cycle_length = 0;
    int repetitions = 0;
    const auto current_hash = board.get_hash();
    const auto size = m_history.size();

    for (int idx = size - 3; idx >= m_startboard; idx -= 2) {
        const auto hash = m_history[idx]->get_hash();
        if (hash == current_hash) {
            cycle_length = size - idx - 1;
            repetitions = 1 + m_history[idx]->get_repetitions();
            break;
        }
    }
    assert(cycle_length % 2 == 0);
    board.set_repetitions(repetitions, cycle_length);
}

int Position::get_repetitions() const {
    return board.get_repetitions();
}

int Position::get_cycle_length() const {
    return board.get_cycle_length();
}

std::array<Types::Vertices, 2> Position::get_kings() const {
    return board.get_kings();
}

bool Position::is_capture() const {
    return board.is_capture();
}

bool Position::is_check(const Types::Color color) const {
    return board.is_check(color);
}

std::string Position::history_board() const {
    auto out = std::ostringstream{};
    auto idx = size_t{0};
    for (const auto &board : m_history) {
        const auto lastmove = board->get_last_move();
        out << "Board Index : " << ++idx << std::endl;
        if (option<bool>("using_traditional_chinese")) {
            board->board_stream<Types::TRADITIONAL_CHINESE>(out, lastmove);
        } else {
            board->board_stream<Types::ASCII>(out, lastmove);
        }
    }
    out << std::endl;
    return out.str();
}

std::vector<std::shared_ptr<const Board>>& Position::get_history() {
    return m_history;
}

std::string Position::get_fen() const {
    auto out = std::ostringstream{};
    board.fen_stream(out);
    return out.str();
}

int Position::get_max_moves() const {
    return m_maxmoves;
}

int Position::get_historysize() const {
    return static_cast<int>(m_history.size());
}

void Position::set_max_moves(int moves) {
    m_maxmoves = moves < 1 ? 1 : moves;
}
