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

#ifndef POSITION_H_INCLUDE
#define POSITION_H_INCLUDE

#include "Board.h"
#include "Types.h"

#include <vector>
#include <memory>
#include <string>

class Position {
public:
    enum class Repetition {
        NONE = 0, DRAW, LOSE, UNKNOWN
    };

    void init_game();

    void display() const;

    void do_resigned();
    void push_board();
    bool fen(std::string &fen);
    bool is_legal(Move move);
    bool do_move(Move move);
    void do_move_assume_legal(Move move);
    bool do_textmove(std::string move);

    bool undo_move();
    bool undo_move(int moves_age);

    bool gameover(bool searching);
    bool position(std::string &fen, std::string &moves);

    std::vector<Move> get_movelist();

    Types::Color get_to_move() const;
    int get_movenum() const;
    int get_gameply() const;
    int get_historysize() const;
    Move get_last_move() const;

    Types::Piece_t get_piece_type(const Types::Vertices vtx) const;
    Types::Piece get_piece(const Types::Vertices vtx) const;
    std::string get_fen() const;
    
    Board board;
    
    Types::Color get_winner(bool searching);
    std::uint64_t get_hash() const;
    std::uint64_t calc_hash() const;

    const std::shared_ptr<const Board> get_past_board(const int p) const;
    std::vector<std::shared_ptr<const Board>>& get_history();
    
    bool is_capture() const;
    bool is_check(const Types::Color color) const;
    std::string history_board() const;

    std::string get_wxfstring(Move m) const;
    int get_repetitions() const;
    int get_cycle_length() const;
    int get_rule50_ply() const;
    int get_rule50_ply_left() const;
    std::array<Types::Vertices, 2> get_kings() const;

    Repetition get_threefold_repetitions_result();
    Move get_forced_checkmate_move(int maxdepth);

private:
    void compute_repetitions();

    Types::Color resigned{Types::INVALID_COLOR};

    int m_startboard;
    bool m_simple_rule;

    std::vector<std::shared_ptr<const Board>> m_history;

};

#endif
