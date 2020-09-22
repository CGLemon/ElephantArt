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

class Board {
public:
    static constexpr int WIDTH = BITBOARD_WIDTH;
    
    static constexpr int HEIGHT = BITBOARD_HEIGHT;

    static constexpr int SHIFT = BITBOARD_SHIFT;

    static constexpr int NUM_VERTICES = SHIFT * HEIGHT;

    enum Piece : int {
        R_PAWN = 0, R_HORSE, R_CANNON, R_ROOK, R_ELEPHANT, R_ADVISOR, R_KING,
        B_PAWN = 7, B_HORSE, B_CANNON, B_ROOK, B_ELEPHANT, B_ADVISOR, B_KING,
        EMPTY_PIECE, INVAL_PIECE, 
        PIECE_NB = 16
    };


    void reset_board();

    void dump_board() const;

    static int get_vertex(const int x, const int y);

    static int get_index(const int x, const int y);

    static int get_x(const int vtx);

    static int get_y(const int vtx);

    static const std::array<Piece, NUM_VERTICES> &get_startvec();

    const std::array<Piece, NUM_VERTICES> &get_boardvec() const;

    bool is_on_board(const int vtx) const;

    void fen_stream(std::ostream &out) const;

    std::uint64_t calc_hash() const;

    static constexpr std::array<Types::Direction, 8> m_dirs =
        {Types::NORTH, Types::EAST, Types::SOUTH, Types::WEST,
        Types::NORTH_EAST, Types::SOUTH_EAST, Types::SOUTH_WEST, Types::NORTH_WEST};


    static void init_mask();

private:
    static constexpr std::array<Piece, NUM_VERTICES> START_VERTICES = {
        R_ROOK,      R_HORSE,     R_ELEPHANT,  R_ADVISOR,   R_KING,      R_ADVISOR,   R_ELEPHANT,  R_HORSE,     R_ROOK,      INVAL_PIECE,
        EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, INVAL_PIECE,
        EMPTY_PIECE, R_CANNON,    EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, R_CANNON,    EMPTY_PIECE, INVAL_PIECE,
        R_PAWN,      EMPTY_PIECE, R_PAWN,      EMPTY_PIECE, R_PAWN,      EMPTY_PIECE, R_PAWN,      EMPTY_PIECE, R_PAWN,      INVAL_PIECE,
        EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, INVAL_PIECE,
                                                         // 楚河  漢界
        EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, INVAL_PIECE,
        B_PAWN,      EMPTY_PIECE, B_PAWN,      EMPTY_PIECE, B_PAWN,      EMPTY_PIECE, B_PAWN,      EMPTY_PIECE, B_PAWN,      INVAL_PIECE,
        EMPTY_PIECE, B_CANNON,    EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, B_CANNON,    EMPTY_PIECE, INVAL_PIECE,
        EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, EMPTY_PIECE, INVAL_PIECE,
        B_ROOK,      B_HORSE,     B_ELEPHANT,  B_ADVISOR,   B_KING,      B_ADVISOR,   B_ELEPHANT,  B_HORSE,     B_ROOK,      INVAL_PIECE
    };

    static std::array<BitBoard, NUM_VERTICES> m_house_mask;

    static std::array<BitBoard, NUM_VERTICES> m_elephant_mask;

    static std::array<BitBoard, NUM_VERTICES> m_advisor_mask;

    static std::array<BitBoard, NUM_VERTICES> m_king_mask;

    std::array<Piece, NUM_VERTICES> m_state;

    std::array<BitBoard, 2> m_bb_color;
    std::array<Types::Vertices, 2> m_king_vertex;
    std::array<std::array<Types::Vertices, 2>, 2> m_advisor_vertex;
    std::array<std::array<Types::Vertices, 2>, 2> m_elephant_vertex;
    std::array<std::array<Types::Vertices, 2>, 2> m_horse_vertex;
    std::array<std::array<Types::Vertices, 2>, 2> m_rook_vertex;
    std::array<std::array<Types::Vertices, 2>, 2> m_cannon_vertex;
    std::array<std::array<Types::Vertices, 5>, 2> m_pawn_vertex;

    Types::Color m_tomove;

    int m_movenum;

    Move m_lastmove;

    std::uint64_t m_hash;

    void generate_king_move(Types::Color color, std::vector<Move> &MoveList) const;

    void piece_stream(std::ostream &out, Piece p) const;

    void piece_stream(std::ostream &out, const int x, const int y) const;

    void info_stream(std::ostream &out) const;

    void board_stream(std::ostream &out) const;
};

inline int Board::get_vertex(const int x, const int y) {
    assert(x >= 0 || x < WIDTH);
    assert(y >= 0 || y < HEIGHT);

    const auto vertex = x + y * SHIFT;
    return vertex;
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
