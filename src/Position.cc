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
#include "ForcedCheckmate.h"
#include "Utils.h"

#include <queue>
#include <iterator>
#include <sstream>

void Position::init_game() {
    m_startboard = 0;
    m_simple_rule = option<bool>("simple_rule");
    m_history.clear();
    board.reset_board();
    push_board();
    resigned = Types::INVALID_COLOR;
}

void Position::display() const {
    const auto lastmove = get_last_move();
    if (option<bool>("using_chinese")) {
        LOGGING << board.get_boardstring<Types::CHINESE>(lastmove);
    } else {
        LOGGING << board.get_boardstring<Types::ASCII>(lastmove);
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
    compute_repetitions();
    push_board();
    if (is_capture()) {
        m_startboard = get_historysize() - 1;
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

bool Position::undo_move() {
    const auto size = get_historysize();
    assert(size >= 1);
    if (size == 1) {
        return false;
    }
    m_history.resize(size-1);
    board = *m_history[size-2];
    return true;
}

bool Position::undo_move(int moves_age) {
    auto success = true;
    for (int i = 0; i < moves_age; ++i) {
        success &= undo_move();
    }
    return success;
}


std::vector<Move> Position::get_movelist() {
    auto movelist = std::vector<Move>{};
    board.generate_movelist(get_to_move(), movelist);

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
    auto moves_success = true;
    auto move_cnt = 0;

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

            if (++move_cnt != (int)chain_moves.size()) {
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
        assert(get_historysize() == move_cnt + 1);
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

    const auto kings = get_kings();
    if (kings[Types::RED] == Types::NO_VERTEX) {
        return Types::BLACK;
    } else if (kings[Types::BLACK] == Types::NO_VERTEX) {
        return Types::RED;
    }

    // According Asian Xiangqi Federation rules, if the current
    // player can eat opponent king. The current player win the
    // game.
    const auto to_move = get_to_move();
    if (!searching && is_check(to_move)) {
        return to_move;
    }

    if (get_rule50_ply_left() <= 0) {
        return Types::EMPTY_COLOR;
    }

    const auto res = get_threefold_repetitions_result();
    if (res == Repetition::DRAW) {
        return Types::EMPTY_COLOR;
    } else if (res == Repetition::LOSE) {
        return to_move;
    } else if (res == Repetition::UNKNOWN) {
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
    return board.get_hash();
}

std::uint64_t Position::calc_hash() const {
    return board.calc_hash();
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
    assert(0 <= p && p <= get_historysize() - 1);
    return m_history[get_historysize() - p - 1];
}

void Position::compute_repetitions() {
    // We don't have push the board into history buffer yet!
    int cycle_length = 0;
    int repetitions = 0;
    bool cutoff = false;

    const auto current_hash = board.get_hash();
    const auto size = get_historysize();

    for (int idx = size - 2; idx >= m_startboard; idx -= 2) {
        const auto hash = m_history[idx]->get_hash();
        if (m_history[idx]->get_repetitions() == 0) {
            cutoff = true;
        }
        if (hash == current_hash) {
            cycle_length = size - idx;
            repetitions = 1 + m_history[idx]->get_repetitions();
            if (cutoff) {
                repetitions = 1;
            }
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

int Position::get_rule50_ply() const {
    return board.get_rule50_ply();
}

int Position::get_rule50_ply_left() const {
    return board.get_rule50_ply_left();
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
        if (option<bool>("using_chinese")) {
            board->board_stream<Types::CHINESE>(out, lastmove);
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

std::string Position::get_wxfstring(Move m) const {
    return board.get_wxfstring(m);
}

std::string Position::get_fen() const {
    auto out = std::ostringstream{};
    board.fen_stream(out);
    return out.str();
}

int Position::get_historysize() const {
    return static_cast<int>(m_history.size());
}

Position::Repetition Position::get_threefold_repetitions_result() {
    // The full asian rule is here.
    // https://www.asianxiangqi.org/%E6%AF%94%E8%B5%9B%E8%A7%84%E4%BE%8B/%E6%AF%94%E8%B5%9B%E8%A7%84%E4%BE%8B_2017.pdf

    if (get_repetitions() < 3) {
        return Repetition::NONE;
    }
    assert(m_history[end_idx - 1]->get_repetitions() == 2);

    auto start_index = 0;
    for (int i = 0; i < get_historysize(); ++i) {
        if (m_history[i]->get_repetitions() == 1) {
            start_index = i;
            break;
        }
    }

    assert(m_history[start_index-1]->get_repetitions() == 0);
    assert(m_history[start_index+get_cycle_length()]->get_repetitions() == 2);
    assert(get_cycle_length() >= 4);
    assert(get_cycle_length() % 2 == 0);

    int curr_ckecking_cnt = 0;
    int other_ckecking_cnt = 0;

    auto curr_bb_pursuit = onBoard;

    for (int i = 0; i < get_cycle_length(); ++i) {
        const auto curr_side  = (i%2 == 0);

        const auto &board = m_history[i + start_index];
        const auto &past_board = m_history[i + start_index-1];
        const auto color = Board::swap_color(board->get_to_move());
        const auto opp_color = board->get_to_move();
        const auto last_move = board->get_last_move();

        if (board->is_check(color)) {
            if (curr_side) {
                ++curr_ckecking_cnt;
            } else {
                ++other_ckecking_cnt;
            }
        }

        const auto bb_colors = board->get_bb_colors();
        const auto bb_attack = board->get_bb_attacks()[color];
        const auto past_bb_attack = past_board->get_bb_attacks()[color];

        // New attack bits after doing a move.
        const auto bb_pursuit = bb_colors[opp_color] &
                                    (bb_attack ^ (bb_attack & past_bb_attack));

        if (!curr_side) {
            curr_bb_pursuit &= bb_pursuit;
        } else {
            if (Utils::on_area(last_move.get_from(), curr_bb_pursuit) && i != 0) {
                // Update pursuit bitboard.
                curr_bb_pursuit ^= Utils::vertex2bitboard(last_move.get_from());
                curr_bb_pursuit ^= Utils::vertex2bitboard(last_move.get_to());
            }
        }
    }

    if (curr_ckecking_cnt == get_cycle_length()/2) {
        if (curr_ckecking_cnt == other_ckecking_cnt) {
            // The case is both sides are the perpetual check. The game is draw.
            return Repetition::DRAW;
        }

        if (curr_ckecking_cnt != 0) {
            return Repetition::DRAW; 
        }

        // The case is the perpetual check. Current player lose the game.
        return Repetition::LOSE;
    }

    if (!curr_bb_pursuit) {
        // No piece is pursued in the every cycle boards. The game is draw.
        return Repetition::DRAW;
    }

    if (m_simple_rule) {
        // The others case is perpetual pursuit, and we simply think all perpetual pursuit is
        // draw.
        return Repetition::DRAW;
    }

    const auto last_move = get_last_move(); 
    const auto pt = board.get_piece_type(last_move.get_to());

    assert(pt != Types::EMPTY_PIECE_T);

    // TODO: Support fully asian rule.
    if (pt == Types::KING ||
            pt == Types::PAWN) {
        // We simply think the last move cause the perpetual pursuit.
        return Repetition::DRAW;
    }

    // We simply think the case is always the perpetual pursuit case. Current player lose the game.
    return Repetition::LOSE;
}

Move Position::get_forced_checkmate_move(int maxdepth) {
    auto forced = ForcedCheckmate(*this);
    return forced.find_checkmate(maxdepth);
}
