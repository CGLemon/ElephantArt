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

#include "Board.h"
#include "Utils.h"
#include "Zobrist.h"

#include <algorithm>
#include <sstream>


constexpr std::array<Board::Piece, Board::NUM_VERTICES> Board::START_VERTICES;

constexpr std::array<Types::Direction, 8> Board::m_dirs;

std::array<BitBoard, Board::NUM_VERTICES> Board::m_house_mask;

std::array<BitBoard, Board::NUM_VERTICES> Board::m_elephant_mask;

std::array<BitBoard, Board::NUM_VERTICES> Board::m_advisor_mask;

std::array<BitBoard, Board::NUM_VERTICES> Board::m_king_mask;

void Board::reset_board() {

    m_tomove = Types::RED;
    m_movenum = 0;

    m_hash = calc_hash();
    m_lastmove = Move();

    for (auto c = 0; c < 2; ++c) {
        m_king_vertex[c] = Types::NO_VERTEX;
        m_advisor_vertex[c].fill(Types::NO_VERTEX);
        m_elephant_vertex[c].fill(Types::NO_VERTEX);
        m_horse_vertex[c].fill(Types::NO_VERTEX);
        m_rook_vertex[c].fill(Types::NO_VERTEX);
        m_cannon_vertex[c].fill(Types::NO_VERTEX);
        m_pawn_vertex[c].fill(Types::NO_VERTEX);
    }


    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const auto vtx = get_vertex(x, y);
            const auto p = static_cast<int>(m_state[vtx]);
            if (p >= 0 && p < 7) {
                m_bb_color[Types::RED] |= BitUtils::vertex2bitboard(Types::Vertices(vtx));
            }
            if (p >= 7 && p < 14) {
                m_bb_color[Types::BLACK] |= BitUtils::vertex2bitboard(Types::Vertices(vtx));
            }
        }
    }
}

void Board::init_mask() {

    // horse mask
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const auto bb = BitUtils::vertex2bitboard(Types::Vertices(v));
        auto mask = BitBoard(0ULL);

        if (BitUtils::on_board(bb)) {
            for (int k = 0; k < 4; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = BitUtils::shift(dir, bb);
                mask |= k_bb;
            }
        }
        m_house_mask[v] = mask;
        // printf("vertex : %d\n", v);
        // BitUtils::dump_bitboard(m_house_mask[v]);
    }

    // elephant mask
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const auto bb = BitUtils::vertex2bitboard(Types::Vertices(v));
        auto mask = BitBoard(0ULL);
        if (BitUtils::on_board(bb)) {
            for (int k = 4; k < 8; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = BitUtils::shift(dir, bb);
                mask |= k_bb;
            }
        }
        m_elephant_mask[v] = mask;
        // printf("vertex : %d\n", v);
        // BitUtils::dump_bitboard(m_elephant_mask[v]);
    }

    // advisor
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const auto bb = BitUtils::vertex2bitboard(Types::Vertices(v));
        auto mask = BitBoard(0ULL);
        if (BitUtils::on_area(bb, KingArea)) {
            for (int k = 4; k < 8; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = BitUtils::shift(dir, bb);
                mask |= k_bb;
            }
            mask &= KingArea;
        }
        m_advisor_mask[v] = mask;
        // printf("vertex : %d\n", v);
        // BitUtils::dump_bitboard(m_advisor_mask[v]);
    }

    // king
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const auto bb = BitUtils::vertex2bitboard(Types::Vertices(v));
        auto mask = BitBoard(0ULL);
        if (BitUtils::on_area(bb, KingArea)) {
            for (int k = 0; k < 4; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = BitUtils::shift(dir, bb);
                mask |= k_bb;
            }
            mask &= KingArea;
        }
        m_king_mask[v] = mask;
        // printf("vertex : %d\n", v);
        // BitUtils::dump_bitboard(m_king_mask[v]);
    }
}

void Board::piece_stream(std::ostream &out, Piece p) const {

    p == R_PAWN      ? out << "P" : p == B_PAWN      ? out << "p" :
    p == R_HORSE     ? out << "N" : p == B_HORSE     ? out << "n" :
    p == R_CANNON    ? out << "C" : p == B_CANNON    ? out << "c" :
    p == R_ROOK      ? out << "R" : p == B_ROOK      ? out << "r" :
    p == R_ELEPHANT  ? out << "B" : p == B_ELEPHANT  ? out << "b" :
    p == R_ADVISOR   ? out << "A" : p == B_ADVISOR   ? out << "a" :
    p == R_KING      ? out << "K" : p == B_KING      ? out << "k" :
    p == EMPTY_PIECE ? out << " " : out << "error";
}

void Board::piece_stream(std::ostream &out, const int x, const int y) const {
    auto p = m_state[get_vertex(x, y)];
    piece_stream(out, p);
}

void Board::info_stream(std::ostream &out) const {

    out << "{";
    if (m_tomove == Types::RED) {
        out << "Next player : RED";
    } else if (m_tomove == Types::BLACK) {
        out << "Next player : BLACK";
    }  else {
        out << "color error!";
    }

    out << ", Last move : ";
    out << m_lastmove.to_string(); 

    out << ", Hash : ";
    out << std::hex;
    out << m_hash;
    out << std::dec;

    out << ",\n Fen : ";
    fen_stream(out);
    out << "}";
    out << std::endl;
}

void Board::fen_stream(std::ostream &out) const {

    for (int y = HEIGHT - 1; y >= 0; --y) {
        auto skip = size_t{0};
        for (int x = 0; x < WIDTH; ++x) {
            const auto vtx = get_vertex(x, y);
            const auto p = m_state[vtx];
            if (p == EMPTY_PIECE) { 
                skip++;
                continue; 
            }

            if (skip != 0) {
                out << skip;
                skip = 0;
            }
            piece_stream(out, p);
        }
        if (skip != 0) {
            out << skip;
        }

        if (y != 0) {
            out << "/";
        }
    }

    out << " ";
    m_tomove == Types::RED ? out << "w" : out << "r";

    out << " - - 0 " << m_movenum+1;
}

void Board::board_stream(std::ostream &out) const {

    for (int y = 0; y < HEIGHT; ++y) {
        Utils::space_stream(out, 1);
        out << "+---+---+---+---+---+---+---+---+---+";
        Utils::strip_stream(out, 1);

        for (int x = 0; x < WIDTH; ++x) {
            out << " | ";

            const auto coordinate_x = WIDTH - x - 1;
            const auto coordinate_y = HEIGHT - y - 1;
            piece_stream(out, coordinate_x, coordinate_y);
        }
        out << " |";
        Utils::strip_stream(out, 1);
    }
    Utils::space_stream(out, 1);
    out << "+---+---+---+---+---+---+---+---+---+";
    Utils::strip_stream(out, 1);

    info_stream(out);
}

std::uint64_t Board::calc_hash() const {

    auto res = Zobrist::zobrist_empty;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const auto vtx = get_vertex(x, y);
            const auto p = static_cast<int>(m_state[vtx]);
            if (is_on_board(vtx)) {
                res ^= Zobrist::zobrist[p][vtx];
            } 
        }
    }
    if (m_tomove == Types::RED) {
        res ^= Zobrist::zobrist_redtomove;
    }
    return res;
}

bool Board::is_on_board(const int vtx) const {
    return m_state[vtx] != INVAL_PIECE;
}

void Board::dump_board() const {
    auto out = std::ostringstream{};
    board_stream(out);
    Utils::auto_printf(out);
}

const std::array<Board::Piece, Board::NUM_VERTICES>& Board::get_startvec() {
    return START_VERTICES;
}

const std::array<Board::Piece, Board::NUM_VERTICES>& Board::get_boardvec() const {
    return m_state;
}

void Board::generate_king_move(Types::Color color, std::vector<Move> &MoveList) const {
    const auto v = m_king_vertex[color];
    // const auto king_bitboard = BitUtils::vertex2bitboard(v);
    const auto mask = m_king_mask[v];

    const auto block_bitboard = mask & m_bb_color[color];
    auto legal_bitboard = mask ^ block_bitboard;

    while(legal_bitboard) {
        const auto res = BitUtils::lsb(legal_bitboard);
        assert(res != -1);

        const auto from = v;
        const auto to = static_cast<Types::Vertices>(res);

        legal_bitboard = BitUtils::reset_ls1b(legal_bitboard);

        MoveList.emplace_back(std::move(Move(from, to)));
    }
}

