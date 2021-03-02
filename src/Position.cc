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

#include <queue>
#include <iterator>
#include <sstream>

void Position::init_game(const int tag) {
    m_drawmoves = 150;
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
    assert(get_gameply() == (int)m_history.size()-1);
}

bool Position::fen(std::string &fen) {
    auto fork_board = std::make_shared<Board>(board);
    auto success = fork_board->fen2board(fen);
    auto current_ply = fork_board->get_gameply();

    if (!success) {
        return false;
    } 

    if (current_ply > get_gameply()) {
        auto fill_board = std::make_shared<Board>(board);
        while ((int)m_history.size() <= current_ply) {
            fill_board->increment_gameply();
            fill_board->swap_to_move();
            m_history.emplace_back(std::make_shared<const Board>(*fill_board));
        }
        assert(fill_board->get_gameply() == (int)m_history.size()-1);
    } else {
        m_history.resize(current_ply+1);
    }

    if (fork_board->get_hash() != m_history[current_ply]->get_hash()) {
        m_history[current_ply] = fork_board;
    }

    m_startboard = current_ply;
    board = *m_history[current_ply];

    return true;
}

bool Position::is_legal(Move move) {
    return board.is_legal(move);
}

void Position::do_move_assume_legal(Move move) {
    board.do_move_assume_legal(move);
    push_board();
    if (is_eaten()) {
        m_startboard = get_gameply();
    }
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

bool Position::undo() {
    const auto ply = get_gameply();
    if (ply == 0) {
        return false;
    }

    m_history.resize(ply);
    board = *m_history[ply - 1];

    assert(get_gameply() == ply-1);
    assert(get_gameply() == (int)m_history.size()-1);

    return true;
}

bool Position::position(std::string &fen, std::string &moves) {
    // First: Set the Fen.
    auto fork_board = std::make_shared<Board>(board);
    auto success = fork_board->fen2board(fen);
    auto current_ply = fork_board->get_gameply();

    if (!success) {
        return false;
    }

    // Second: Do moves.
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

        current_ply += move_cnt;
        assert(get_gameply() == current_ply);
        assert(get_gameply() == (int)m_history.size()-1);
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
    if (!searching && is_checkmate(to_move)) {
        return to_move;
    }

    const auto kings = get_kings();
    if (kings[Types::RED] == Types::NO_VERTEX) {
        return Types::BLACK;
    } else if (kings[Types::BLACK] == Types::NO_VERTEX) {
        return Types::RED;
    }

    if (get_movenum() > m_drawmoves) {
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
    const auto ply = get_gameply();
    assert(0 <= p && p <= ply);
    return m_history[ply - p];
}

std::pair<int, int> Position::get_repetitions() const {
    static constexpr auto MIN_REPETITIONS_CNT = 4; 
    const auto endboard = get_gameply();
    const auto startboard = m_startboard;
    const auto length = endboard - startboard + 1;

    assert(length >= 1);
    if (length <= 2 * MIN_REPETITIONS_CNT - 1) {
        return std::make_pair(0, MIN_REPETITIONS_CNT);
    }

    const auto buffer_size = length/2;
    auto boardhash = std::vector<std::uint64_t>(length);
    auto buffer = std::vector<std::uint64_t>(buffer_size);
    auto repetitions = 0;
    assert(buffer_size >= MIN_REPETITIONS_CNT);

    for (int i = 0; i < length; ++i) {
        boardhash[i] = m_history[endboard-i]->get_hash();
    }
    std::copy(std::begin(boardhash),
                  std::begin(boardhash) + buffer_size,
                  std::begin(buffer));

    const auto repetitions_proccess = [](std::vector<std::uint64_t> &bd_hash,
                                         std::vector<std::uint64_t> &bf,
                                         const int offset, const int cnt) -> bool {
        for (int i = 0; i < cnt; ++i) {
            if (bd_hash[i + offset] != bf[i]) {
                return false;
            }
        }
        return true;
    };

    int repetitions_cnt = MIN_REPETITIONS_CNT;
    for (; repetitions_cnt <= buffer_size; ++repetitions_cnt) {
        for (int offset = repetitions_cnt; offset + repetitions_cnt < length; offset += repetitions_cnt) {
            const auto success = repetitions_proccess(boardhash, buffer, offset, repetitions_cnt);
            if (success) {
                repetitions++;
            } else {
                break;
            }
        }
        if (repetitions > 0) {
            break;
        }
    }

    assert(repetitions <= 2);
    return std::make_pair(repetitions, repetitions_cnt);
}

std::array<Types::Vertices, 2> Position::get_kings() const {
    return board.get_kings();
}

bool Position::is_eaten() const {
    return board.is_eaten();
}

bool Position::is_checkmate(const Types::Color color) const {
    return board.is_checkmate(color);
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

std::string Position::get_wxfmove() const {
    return board.get_wxfmove();
}

int Position::get_draw_moves() const {
    return m_drawmoves;
}

void Position::set_draw_moves(int moves) {
    m_drawmoves = moves < 1 ? 1 : moves;
}
