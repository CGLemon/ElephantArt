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

#ifndef POSITION_H_INCLUDE
#define POSITION_H_INCLUDE

#include "Board.h"

#include <vector>
#include <memory>
#include <string>

class Position {
public:
    void init_game();

    void display() const;

    void push_board();
    bool fen2board(std::string &fen);
    bool is_legal(Move move) const;
    bool do_move(Move move);
    void do_move_assume_legal(Move move);
    bool do_textmove(std::string move);

    bool undo();

    bool position(std::string &fen, std::string &moves);

    std::vector<Move> get_movelist() const;
    Types::Color get_to_move() const;
    int get_movenum() const;
    std::uint64_t get_hash() const;
    Move get_last_move() const;

    Board board;

private:
    std::vector<std::shared_ptr<const Board>> m_history;

};

#endif
