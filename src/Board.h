/*
    This file is part of Saya.
    Copyright (C) 2020 Hung-Zhe, Lin

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

#ifndef BOARD_H_INCLUDE
#define BOARD_H_INCLUDE

#include "Uint128_t.h"
#include "BitBoard.h"

#include <cassert>
#include <array>
#include <vector>
#include <iostream>
#include <string>

class Board {
public:
    static constexpr int WIDTH = BITBOARD_WIDTH;
    
    static constexpr int HEIGHT = BITBOARD_HEIGHT;

    static constexpr int SHIFT = BITBOARD_SHIFT;

    static constexpr int NUM_VERTICES = SHIFT * HEIGHT;

    void reset_board();

    void dump_board() const;

    static int get_vertex(const int x, const int y);

    static int get_index(const int x, const int y);

    static Types::Color swap_color(const Types::Color color);

    static int get_x(const int vtx);

    static int get_y(const int vtx);

    static std::string get_start_position();

    Types::Piece get_piece(const int x, const int y) const;

    Types::Piece get_piece(const int vtx) const;

    Types::Color get_to_move() const;

    void generate_movelist(Types::Color color, std::vector<Move> &MoveList) const;

    bool is_on_board(const int vtx) const;

    void fen_stream(std::ostream &out) const;

    void fen2board(std::string &fen);

    std::uint64_t calc_hash() const;

    static constexpr std::array<Types::Direction, 8> m_dirs =
        {Types::NORTH,      Types::EAST,       Types::SOUTH,      Types::WEST,
         Types::NORTH_EAST, Types::SOUTH_EAST, Types::SOUTH_WEST, Types::NORTH_WEST};


    static void init_mask();

private:
    #define P_  Types::R_PAWN
    #define H_  Types::R_HORSE
    #define C_  Types::R_CANNON
    #define R_  Types::R_ROOK
    #define E_  Types::R_ELEPHANT
    #define A_  Types::R_ADVISOR
    #define K_  Types::R_KING

    #define p_  Types::B_PAWN
    #define h_  Types::B_HORSE
    #define c_  Types::B_CANNON
    #define r_  Types::B_ROOK
    #define e_  Types::B_ELEPHANT
    #define a_  Types::B_ADVISOR
    #define k_  Types::B_KING

    #define ET    Types::EMPTY_PIECE
    #define invalid_  Types::INVAL_PIECE

    static constexpr std::array<Types::Piece, NUM_VERTICES> START_VERTICES = {
        R_, H_, E_, A_, K_, A_, E_, H_, R_, invalid_,
        ET, ET, ET, ET, ET, ET, ET, ET, ET, invalid_,
        ET, C_, ET, ET, ET, ET, ET, C_, ET, invalid_,
        P_, ET, P_, ET, P_, ET, P_, ET, P_, invalid_,
        ET, ET, ET, ET, ET, ET, ET, ET, ET, invalid_,
                 // 楚河  漢界
        ET, ET, ET, ET, ET, ET, ET, ET, ET, invalid_,
        p_, ET, p_, ET, p_, ET, p_, ET, p_, invalid_,
        ET, c_, ET, ET, ET, ET, ET, c_, ET, invalid_,
        ET, ET, ET, ET, ET, ET, ET, ET, ET, invalid_,
        r_, h_, e_, a_, k_, a_, e_, h_, r_, invalid_,
    };

    #undef P_
    #undef H_
    #undef C_
    #undef R_
    #undef E_
    #undef A_
    #undef K_

    #undef p_
    #undef h_
    #undef c_
    #undef r_
    #undef e_
    #undef a_
    #undef k_

    #undef ET
    #undef invalid_

    static std::array<std::array<BitBoard, NUM_VERTICES>, 2> m_pawn_attack;

    static std::array<BitBoard, NUM_VERTICES> m_house_mask;

    static std::array<BitBoard, NUM_VERTICES> m_elephant_mask;

    static std::array<BitBoard, NUM_VERTICES> m_advisor_attack;

    static std::array<BitBoard, NUM_VERTICES> m_king_mask;

    static void init_pawn_attack();
    
    std::array<BitBoard, 2> m_bb_color;

    BitBoard m_bb_pawn;
    BitBoard m_bb_horse;
    BitBoard m_bb_rook;
    BitBoard m_bb_elephant;
    BitBoard m_bb_advisor;
    BitBoard m_bb_cannon;

    std::array<Types::Vertices, 2> m_king_vertex;

    Types::Color m_tomove;

    int m_movenum;

    Move m_lastmove;

    std::uint64_t m_hash;

    template<Types::Piece_t>
    void generate_move(Types::Color color, std::vector<Move> &MoveList) const;

    void generate_king_move(Types::Color color, std::vector<Move> &MoveList) const;

    void generate_pawn_move(Types::Color color, std::vector<Move> &MoveList) const;

    void generate_advisor_move(Types::Color color, std::vector<Move> &MoveList) const;

    void piece_stream(std::ostream &out, Types::Piece p) const;

    void piece_stream(std::ostream &out, const int x, const int y) const;

    void info_stream(std::ostream &out) const;

    void board_stream(std::ostream &out) const;
};

inline int Board::get_vertex(const int x, const int y) {
    assert(x >= 0 || x < WIDTH);
    assert(y >= 0 || y < HEIGHT);

    return x + y * SHIFT;
}

inline int Board::get_index(const int x, const int y) {
    assert(x >= 0 || x < WIDTH);
    assert(y >= 0 || y < HEIGHT);

    return x + y * WIDTH;
}

inline int Board::get_x(const int vertex) {
    return vertex % SHIFT;
}

inline int Board::get_y(const int vertex) {
    return vertex / SHIFT;
}

#endif
