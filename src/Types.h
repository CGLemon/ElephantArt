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

#ifndef TYPE_H_INCLUDE
#define TYPE_H_INCLUDE

#include "config.h"
#include <cstdint>

#define ENABLE_BASE_OPERATORS_ON(T)                                       \
friend constexpr T operator+(T d1, int d2) { return T(int(d1) + d2); }    \
friend constexpr T operator-(T d1, int d2) { return T(int(d1) - d2); }    \
friend constexpr T operator-(T d) { return T(-int(d)); }                  \
friend inline T& operator+=(T& d1, int d2) { return d1 = d1 + d2; }       \
friend inline T& operator-=(T& d1, int d2) { return d1 = d1 - d2; }       \


#define ENABLE_INCR_OPERATORS_ON(T)                                       \
friend inline T& operator++(T& d) { return d = T(int(d) + 1); }           \
friend inline T& operator--(T& d) { return d = T(int(d) - 1); }


#define ENABLE_FULL_OPERATORS_ON(T)                                       \
ENABLE_BASE_OPERATORS_ON(T)                                               \
friend constexpr T operator*(int i, T d) { return T(i * int(d)); }        \
friend constexpr T operator*(T d, int i) { return T(int(d) * i); }        \
friend constexpr T operator/(T d, int i) { return T(int(d) / i); }        \
friend constexpr int operator/(T d1, T d2) { return int(d1) / int(d2); }  \
friend inline T& operator*=(T& d, int i) { return d = T(int(d) * i); }    \
friend inline T& operator/=(T& d, int i) { return d = T(int(d) / i); }

class Types {
public:
    enum Direction : int {
        NORTH =  MARCRO_SHIFT,
        EAST  =  1,
        SOUTH = -NORTH,
        WEST  = -EAST,

        NORTH_EAST = NORTH + EAST,
        SOUTH_EAST = SOUTH + EAST,
        SOUTH_WEST = SOUTH + WEST,
        NORTH_WEST = NORTH + WEST
    };

    ENABLE_BASE_OPERATORS_ON(Direction)

    enum Color : int {
        RED = 0, BLACK, EMPTY_COLOR, INVALID_COLOR, COLOR_NB = 2
    };

    enum Piece_t : int {
        PAWN = 0, CANNON, ROOK, HORSE, ELEPHANT, ADVISOR, KING, EMPTY_PIECE_T, PIECE_T_NB = 7
    };

    ENABLE_BASE_OPERATORS_ON(Piece_t)

    enum Piece : int {
        R_PAWN = 0, R_CANNON, R_ROOK, R_HORSE, R_ELEPHANT, R_ADVISOR, R_KING,
        B_PAWN = 7, B_CANNON, B_ROOK, B_HORSE, B_ELEPHANT, B_ADVISOR, B_KING,
        EMPTY_PIECE,
        INVAL_PIECE, 
        PIECE_NB = 16
    };

    ENABLE_BASE_OPERATORS_ON(Piece)

    enum Vertices : int {
                                                                            // invalid
    VTX_A0 = 0, VTX_B0, VTX_C0, VTX_D0, VTX_E0, VTX_F0, VTX_G0, VTX_H0, VTX_I0, VTX_J0,
        VTX_A1, VTX_B1, VTX_C1, VTX_D1, VTX_E1, VTX_F1, VTX_G1, VTX_H1, VTX_I1, VTX_J1,
        VTX_A2, VTX_B2, VTX_C2, VTX_D2, VTX_E2, VTX_F2, VTX_G2, VTX_H2, VTX_I2, VTX_J2,
        VTX_A3, VTX_B3, VTX_C3, VTX_D3, VTX_E3, VTX_F3, VTX_G3, VTX_H3, VTX_I3, VTX_J3,
        VTX_A4, VTX_B4, VTX_C4, VTX_D4, VTX_E4, VTX_F4, VTX_G4, VTX_H4, VTX_I4, VTX_J4,
        VTX_A5, VTX_B5, VTX_C5, VTX_D5, VTX_E5, VTX_F5, VTX_G5, VTX_H5, VTX_I5, VTX_J5,
        VTX_A6, VTX_B6, VTX_C6, VTX_D6, VTX_E6, VTX_F6, VTX_G6, VTX_H6, VTX_I6, VTX_J6,
        VTX_A7, VTX_B7, VTX_C7, VTX_D7, VTX_E7, VTX_F7, VTX_G7, VTX_H7, VTX_I7, VTX_J7,
        VTX_A8, VTX_B8, VTX_C8, VTX_D8, VTX_E8, VTX_F8, VTX_G8, VTX_H8, VTX_I8, VTX_J8,
        VTX_A9, VTX_B9, VTX_C9, VTX_D9, VTX_E9, VTX_F9, VTX_G9, VTX_H9, VTX_I9, VTX_J9,

        NO_VERTEX = 100,

        VTX_BEGIN = VTX_A0,
        VTX_END   = NO_VERTEX
    };

    ENABLE_FULL_OPERATORS_ON(Vertices)
    ENABLE_INCR_OPERATORS_ON(Vertices)

    enum File : int {
                                                                                // invalid
        FILE_A = 0, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H, FILE_I, FILE_J, FILE_NB
    };

    enum Rank : int {
        RANK_0 = 0, RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8, RANK_9, RANK_NB
    };

    ENABLE_INCR_OPERATORS_ON(File)
    ENABLE_INCR_OPERATORS_ON(Rank)

    enum Language : int {
        ASCII = 0, TRADITIONAL_CHINESE
    };
};

#undef ENABLE_BASE_OPERATORS_ON
#undef ENABLE_INCR_OPERATORS_ON
#undef ENABLE_FULL_OPERATORS_ON

#endif
