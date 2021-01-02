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

#ifndef BITBOARD_H_INCLUDE
#define BITBOARD_H_INCLUDE

#include "Uint128_t.h"
#include "Types.h"
#include "config.h"

#include <iostream>
#include <sstream>
#include <cstdint>
#include <cassert>
#include <string>

typedef Uint128_t BitBoard;

static constexpr int BITBOARD_WIDTH = MARCRO_WIDTH;
static constexpr int BITBOARD_HEIGHT = MARCRO_HEIGHT;
static constexpr int BITBOARD_SHIFT = MARCRO_SHIFT;
static constexpr int BITBOARD_NUM_VERTICES = BITBOARD_SHIFT * MARCRO_HEIGHT;
static constexpr int BITBOARD_INTERSECTIONS = MARCRO_WIDTH * MARCRO_HEIGHT;

const BitBoard FirstPosition(0ULL, 1ULL);

const BitBoard onBoard(0x7fdff7fdf, 0xf7fdff7fdff7fdff);

const BitBoard FileABB(0x4010040, 0x1004010040100401);
const BitBoard FileBBB = FileABB << 1;
const BitBoard FileCBB = FileABB << 2;
const BitBoard FileDBB = FileABB << 3;
const BitBoard FileEBB = FileABB << 4;
const BitBoard FileFBB = FileABB << 5;
const BitBoard FileGBB = FileABB << 6;
const BitBoard FileHBB = FileABB << 7;
const BitBoard FileIBB = FileABB << 8;
const BitBoard FileJBB = FileABB << 9; // invalid

const BitBoard Rank0BB(0x0, 0x1ff);
const BitBoard Rank1BB = Rank0BB << (BITBOARD_SHIFT * 1);
const BitBoard Rank2BB = Rank0BB << (BITBOARD_SHIFT * 2);
const BitBoard Rank3BB = Rank0BB << (BITBOARD_SHIFT * 3);
const BitBoard Rank4BB = Rank0BB << (BITBOARD_SHIFT * 4);
const BitBoard Rank5BB = Rank0BB << (BITBOARD_SHIFT * 5);
const BitBoard Rank6BB = Rank0BB << (BITBOARD_SHIFT * 6);
const BitBoard Rank7BB = Rank0BB << (BITBOARD_SHIFT * 7);
const BitBoard Rank8BB = Rank0BB << (BITBOARD_SHIFT * 8);
const BitBoard Rank9BB = Rank0BB << (BITBOARD_SHIFT * 9);

const BitBoard Square = onBoard | FileJBB;
const BitBoard RedSide = Rank0BB | Rank1BB | Rank2BB | Rank3BB | Rank4BB;
const BitBoard BlackSide = Rank5BB | Rank6BB | Rank7BB | Rank8BB | Rank9BB;
const BitBoard KingArea = (Rank0BB | Rank1BB | Rank2BB | Rank7BB | Rank8BB | Rank9BB) & (FileDBB | FileEBB | FileFBB);

namespace Utils {

inline static bool on_board(const BitBoard bitboard) {
    return onBoard & bitboard;
}

inline static bool on_board(const Types::Vertices v) {
    return onBoard & (FirstPosition << v);
}

inline static bool on_board(const int v) {
    return onBoard & (FirstPosition << v);
}

inline static bool on_area(const BitBoard bitboard, const BitBoard area_board) {
    return area_board & bitboard;
}

inline static bool on_area(const Types::Vertices v, const BitBoard area_board) {
    return area_board & (FirstPosition << v);
}

inline static bool on_area(const int v, const BitBoard area_board) {
    return area_board & (FirstPosition << v);
}

inline static BitBoard shift(Types::Direction d, BitBoard bitboard) {
    if (d > 0) {
        return (bitboard << d) & onBoard;
    }
    return (bitboard >> (-d)) & onBoard;
}

inline static BitBoard file2bitboard(const Types::File f) {
    return FileABB << f; 
}

inline static BitBoard rank2bitboard(const Types::Rank r) {
    return Rank0BB << (BITBOARD_SHIFT * r);
}

inline static BitBoard vertex2bitboard(const Types::Vertices v) {
    return FirstPosition << v;
}

inline static BitBoard vertex2bitboard(const int v) {
    return FirstPosition << v;
}

inline static BitBoard ls1b(BitBoard b) {
    return b & -b;
}

inline static BitBoard reset_ls1b(BitBoard b) {
    return b & (b-1);
}

inline static Types::Vertices lsb(BitBoard b) {

    /*
     * bitScanForward
     * @author Martin Läuter (1997)
     *         Charles E. Leiserson
     *         Harald Prokop
     *         Keith H. Randall
     * "Using de Bruijn Sequences to Index a 1 in a Computer Word"
     * @param bb bitboard to scan
     * @precondition bb != 0
     * @return index (0..63) of least significant one bit
     */
    static constexpr int index64[64] = {
         0,  1, 48,  2, 57, 49, 28,  3,
        61, 58, 50, 42, 38, 29, 17,  4,
        62, 55, 59, 36, 53, 51, 43, 22,
        45, 39, 33, 30, 24, 18, 12,  5,
        63, 47, 56, 27, 60, 41, 37, 16,
        54, 35, 52, 21, 44, 32, 23, 11,
        46, 26, 40, 15, 34, 20, 31, 10,
        25, 14, 19,  9, 13,  8,  7,  6
    };

    const auto bitScanForward = [](std::uint64_t bit64) -> int {
        static constexpr std::uint64_t debruijn64 = 0x03f79d71b4cb0a89;
        return index64[((bit64 & -bit64) * debruijn64) >> 58];
    };

    int res = 0;
    std::uint64_t bit = b.get_lower();
    if (bit == 0ULL) {
        bit = b.get_upper();
        res += 64;
    }

    if (bit == 0ULL) {
        return Types::NO_VERTEX;
    }

    res += bitScanForward(bit);

    return static_cast<Types::Vertices>(res);
}
   
// Counts the number of set bits in the BitBoard.
inline static int count(BitBoard b) {

    std::uint64_t x_1 = b.get_upper();
    std::uint64_t x_2 = b.get_lower();

    const auto lambda_uint64_count = [](std::uint64_t x) -> int {
        x -= (x >> 1) & 0x5555555555555555;
        x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333);
        x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F;
        return (x * 0x0101010101010101) >> 56;
    };

    return lambda_uint64_count(x_1) + lambda_uint64_count(x_2);
}

/*
 * Like count(BitBoard b) but using algorithm faster on a very sparse BitBoard.
 * May be slower for more than 4 set bits, but still correct.
 * Useful when counting bits in a Q, R, N or B BitBoard.
 */
inline static int count_few(BitBoard b) {

    std::uint64_t x_1 = b.get_upper();
    std::uint64_t x_2 = b.get_lower();
    const auto lambda_uint64_count_few = [](std::uint64_t x) -> int {
        int count;
        for (count = 0; x != 0; ++count) {
              x &= x - 1;
        }
        return count;
    };

    return lambda_uint64_count_few(x_1) + lambda_uint64_count_few(x_2);
}

inline static bool exist(BitBoard b, Types::Vertices v) {
    return b == vertex2bitboard(v);
}

inline static Types::Vertices extract(BitBoard &b) {
    const auto vtx =lsb(b);
    assert(vtx != Types::NO_VERTEX);
    b = reset_ls1b(b);
    return vtx;
}
/*
 * Display the bitboard (include invalid edge and extra bits).
 */
void dump_bitboard(const BitBoard &bitboard, std::ostream &out);

void dump_bitboard(const BitBoard &bitboard);

}

class Move {
public:
    Move() = default;

    constexpr Move(const Types::Vertices from_, const Types::Vertices to_) :
                       m_data(static_cast<std::uint16_t>(to_) + (static_cast<std::uint16_t>(from_) << 8)) {}

    Types::Vertices get_from() const;

    Types::Vertices get_to() const;

    std::uint16_t get_data() const;

    BitBoard get_from_bitboard() const {return Utils::vertex2bitboard(get_from()); }

    BitBoard get_to_bitboard() const {return Utils::vertex2bitboard(get_to()); }

    bool hit(BitBoard &b) const;

    bool valid() const;

    bool is_ok() const;

    std::string to_string() const;

    static constexpr std::uint16_t INVALID = 0;

private:
    std::uint16_t m_data{INVALID};

    static constexpr std::uint16_t TO_MASK = 0x00ff;
    static constexpr std::uint16_t FROM_MASK = 0xff00;
};

#endif
