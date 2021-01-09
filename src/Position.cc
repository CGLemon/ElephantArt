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
#include "Zobrist.h"

#include <iterator>
#include <sstream>


void Position::init_game(int tag) {
    m_startboard = 0;
    position_hash = Zobrist::zobrist_positions[tag];
    m_history.clear();
    board.reset_board();
    push_board();
    resigned = Types::INVALID_COLOR;
}

void Position::display() const {
    if (option<bool>("using_traditional_chinese")) {
        board.dump_board<Types::TRADITIONAL_CHINESE>();
    } else {
        board.dump_board<Types::ASCII>();
    }
}

void Position::do_resigned() {
    assert(resigned == Types::INVALID_COLOR);
    resigned = get_to_move();
}

void Position::push_board() {
    m_history.emplace_back(std::make_shared<const Board>(board));
    assert(get_movenum() == (int)m_history.size() - 1);
}

bool Position::fen(std::string &fen) {

    auto fork_board = std::make_shared<Board>(board);
    auto success = fork_board->fen2board(fen);
    auto current_movenum = fork_board->get_movenum();
 
    if (!success) {
        return false;
    } 

    if (current_movenum > get_movenum()) {
        auto fill_board = std::make_shared<Board>(board);
        while ((int)m_history.size() <= current_movenum) {
            fill_board->increment_movenum();
            fill_board->swap_to_move();
            m_history.emplace_back(std::make_shared<const Board>(*fill_board));
            assert(fill_board->get_movenum() == (int)m_history.size() - 1);
        }
    } else {
        m_history.resize(current_movenum-1);
    }

    m_history[current_movenum] = fork_board;
    m_startboard = current_movenum;
    if (board.get_hash() != m_history[current_movenum]->get_hash()) {
        board = *m_history[current_movenum];
    }
    return true;
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
    if (movenum == 0) {
        return false;
    }

    m_history.resize(movenum);
    board = *m_history[movenum - 1];

    assert(get_movenum() == movenum);
    assert(get_movenum() == (int)m_history.size()-1);

    return true;
}

bool Position::position(std::string &fen, std::string& moves) {

    // first : Set the fen.
    auto fork_board = std::make_shared<Board>(board);
    auto success = fork_board->fen2board(fen);
    auto current_movenum = fork_board->get_movenum();

    if (!success) {
        return false;
    }

    // second : Do moves.
    auto chain_board = std::vector<std::shared_ptr<const Board>>{};
    bool moves_success = true;
    auto move_cnt = size_t{0};

    if (!moves.empty()) {
        auto moves_stream = std::stringstream{moves};
        auto move_str = std::string{};

        while (moves_stream >> move_str) {
            const auto move = Board::text2move(move_str);
            ++move_cnt;
            if (move.valid()) {
                if (fork_board->is_legal(move)) {
                    fork_board->do_move(move);
                    chain_board.emplace_back(std::make_shared<Board>(*fork_board));
                }
            }

            if (move_cnt != chain_board.size()) {
                moves_success = false;
                break;
            }
        }
    }

    if (moves_success) {
        Position::fen(fen);
        for (auto i = size_t{0}; i < move_cnt; ++i) {
            m_history.emplace_back(chain_board[i]);
            assert(chain_board[i]->get_movenum() == (int)m_history.size()-1);
        }

        current_movenum += move_cnt;
        m_startboard = current_movenum;
        board = *m_history[current_movenum];
        assert(get_movenum() == current_movenum);
        assert(get_movenum() == (int)m_history.size()-1);
    }

    return moves_success;
}

bool Position::gameover() {
    return get_winner() != Types::INVALID_COLOR;
}

Types::Color Position::get_winner() const {
    if (resigned != Types::INVALID_COLOR) {
        if (resigned == Types::EMPTY_COLOR) {
            return Types::EMPTY_COLOR;
        }
        return Board::swap_color(resigned);
    }

    const auto kings = board.get_kings();
    if (kings[Types::RED] == Types::NO_VERTEX) {
        return Types::BLACK;
    } else if (kings[Types::BLACK] == Types::NO_VERTEX) {
        return Types::RED;
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

std::uint64_t Position::calc_hash(const int symmetry) const {
    return board.calc_hash(symmetry) ^ position_hash;
}

Move Position::get_last_move() const {
    return board.get_last_move();
}

Types::Piece_t Position::get_piece_type(const Types::Vertices vtx) const {
    return board.get_piece_type(vtx);
}

const std::shared_ptr<const Board> Position::get_past_board(const int p) const {
    const auto movenum = get_movenum();
    assert(0 <= p && p <= movenum);
    return m_history[movenum - p];
}

std::string Position::history_board() const {
    auto out = std::ostringstream{};
    for (const auto &board : m_history) {
        if (option<bool>("using_traditional_chinese")) {
            board->board_stream<Types::TRADITIONAL_CHINESE>(out);
        } else {
            board->board_stream<Types::ASCII>(out);
        }
    }
    return out.str();
}
