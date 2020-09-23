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

    auto start_position = get_start_position();
    fen2board(start_position);

    m_hash = calc_hash();
    m_lastmove = Move();
}

void Board::fen2board(std::string &fen) {

    auto king_vertex_black = Types::NO_VERTEX;
    auto king_vertex_red = Types::NO_VERTEX;     

    auto bb_black = ZeroBB;
    auto bb_red = ZeroBB;

    auto bb_pawn = ZeroBB;
    auto bb_horse = ZeroBB;
    auto bb_rook = ZeroBB;
    auto bb_elephant = ZeroBB;
    auto bb_advisor = ZeroBB;
    auto bb_cannon = ZeroBB;


    auto fen_format = std::stringstream{fen};
    auto fen_stream = std::string{};

    fen_format >> fen_stream;

    auto vtx = Types::VTX_A9;
    for (const auto &c : fen_stream) {
        bool skip = false;
        if (c == 'p') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_pawn |= bb;
            bb_black |= bb;
        } else if (c == 'c') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_cannon |= bb;
            bb_black |= bb;
        } else if (c == 'r') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_rook |= bb;
            bb_black |= bb;
        } else if (c == 'n') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_horse |= bb;
            bb_black |= bb;
        } else if (c == 'b') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_elephant |= bb;
            bb_black |= bb;
        } else if (c == 'a') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_advisor |= bb;
            bb_black |= bb;
        } else if (c == 'k') {
            king_vertex_black = vtx;
            bb_black |= BitUtils::vertex2bitboard(vtx);
        } else if (c == 'P') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_pawn |= bb;
            bb_red |= bb;
        } else if (c == 'C') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_cannon |= bb;
            bb_red|= bb;
        } else if (c == 'R') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_rook |= bb;
            bb_red |= bb;
        } else if (c == 'N') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_horse |= bb;
            bb_red |= bb;
        } else if (c == 'B') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_elephant |= bb;
            bb_red |= bb;
        } else if (c == 'A') {
            auto bb = BitUtils::vertex2bitboard(vtx);
            bb_advisor |= bb;
            bb_red |= bb;
        } else if (c == 'K') {
            king_vertex_red = vtx;
            bb_red |= BitUtils::vertex2bitboard(vtx);
        } else if (c >= '1' && c <= '9') {
            vtx += (std::atoi(&c) - 1);
        } else if (c == '/') {
            assert(!is_on_board(vtx));
            vtx -= (2 * SHIFT - 1);
            skip = true;
        }

        if (vtx == Types::VTX_J0) {
            break;
        }
        if (!skip) {
            ++vtx;
        }
    }

    fen_format >> fen_stream;
    if (fen_stream == "w" || fen_stream == "r") {
        m_tomove = Types::RED;
    } else if (fen_stream == "b") {
        m_tomove = Types::BLACK;
    }


    m_king_vertex[Types::RED] = king_vertex_red;
    m_king_vertex[Types::BLACK] = king_vertex_black;    

    m_bb_color[Types::RED] = bb_red;
    m_bb_color[Types::BLACK] = bb_black;

    m_bb_pawn = bb_pawn;
    m_bb_horse = bb_horse;
    m_bb_rook = bb_rook;
    m_bb_elephant = bb_elephant;
    m_bb_advisor = bb_advisor;
    m_bb_cannon = bb_cannon;
}

void Board::init_mask() {

    // horse mask
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const auto bb = BitUtils::vertex2bitboard(v);
        auto mask = BitBoard(0ULL);

        if (BitUtils::on_board(bb)) {
            for (int k = 0; k < 4; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = BitUtils::shift(dir, bb);
                mask |= k_bb;
            }
        }
        m_house_mask[v] = mask;
    }

    // elephant mask
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const auto bb = BitUtils::vertex2bitboard(v);
        auto mask = BitBoard(0ULL);
        if (BitUtils::on_board(bb)) {
            for (int k = 4; k < 8; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = BitUtils::shift(dir, bb);
                mask |= k_bb;
            }
        }
        m_elephant_mask[v] = mask;
    }

    // advisor
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const auto bb = BitUtils::vertex2bitboard(v);
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
    }

    // king
    for (int v = 0; v < NUM_VERTICES; ++v) {
        const auto bb = BitUtils::vertex2bitboard(v);
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
    auto p = get_piece(x, y);
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
            const auto pis = get_piece(vtx);
            if (pis == EMPTY_PIECE) { 
                skip++;
                continue; 
            }

            if (skip != 0) {
                out << skip;
                skip = 0;
            }
            piece_stream(out, pis);
        }
        if (skip != 0) {
            out << skip;
        }

        if (y != 0) {
            out << "/";
        }
    }

    out << " ";
    m_tomove == Types::RED ? out << "w" : out << "b";

    out << " - - 0 " << m_movenum+1;
}

void Board::board_stream(std::ostream &out) const {

    for (int y = 0; y < HEIGHT; ++y) {
        Utils::space_stream(out, 1);
        out << "+---+---+---+---+---+---+---+---+---+";
        Utils::strip_stream(out, 1);

        for (int x = 0; x < WIDTH; ++x) {
            out << " | ";

            const auto coordinate_x = x;
            const auto coordinate_y = HEIGHT - y - 1;
            piece_stream(out, coordinate_x, coordinate_y);
        }
        out << " | ";
        out << HEIGHT - y - 1;
        Utils::strip_stream(out, 1);
    }
    Utils::space_stream(out, 1);
    out << "+---+---+---+---+---+---+---+---+---+" << std::endl;
    out << "   a   b   c   d   e   f   g   h   i";
    Utils::strip_stream(out, 1);

    info_stream(out);
}

std::uint64_t Board::calc_hash() const {

    auto res = Zobrist::zobrist_empty;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const auto vtx = get_vertex(x, y);
            const auto pis = get_piece(vtx);
            if (is_on_board(vtx)) {
                res ^= Zobrist::zobrist[pis][vtx];
            } 
        }
    }
    if (m_tomove == Types::RED) {
        res ^= Zobrist::zobrist_redtomove;
    }
    return res;
}

bool Board::is_on_board(const int vtx) const {
    return START_VERTICES[vtx] != INVAL_PIECE;
}

void Board::dump_board() const {
    auto out = std::ostringstream{};
    board_stream(out);
    Utils::auto_printf(out);
}

std::string Board::get_start_position() {
    return std::string{"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"};
}


Board::Piece Board::get_piece(const int x, const int y) const {
    return get_piece(get_vertex(x, y));
}


Board::Piece Board::get_piece(const int vtx) const {

    auto color = Types::COLOR_INVALID;
    auto p = INVAL_PIECE;

    if (BitUtils::on_area(vtx, m_bb_color[Types::RED])) {
        color = Types::RED;
    } else if (BitUtils::on_area(vtx, m_bb_color[Types::BLACK])) {
        color = Types::BLACK;
    }

    if (color == Types::COLOR_INVALID) {
        if (is_on_board(vtx)) {
            return EMPTY_PIECE;
        } else {
            return INVAL_PIECE;
        }
    }

    if (BitUtils::on_area(vtx, m_bb_pawn)) {
        p = R_PAWN;
    } else if (BitUtils::on_area(vtx, m_bb_horse)) {
        p = R_HORSE;
    } else if (BitUtils::on_area(vtx, m_bb_cannon)) {
        p = R_CANNON;
    } else if (BitUtils::on_area(vtx, m_bb_rook)) {
        p = R_ROOK;
    } else if (BitUtils::on_area(vtx, m_bb_elephant)) {
        p = R_ELEPHANT;
    } else if (BitUtils::on_area(vtx, m_bb_advisor)) {
        p = R_ADVISOR;
    } else {
        assert(vtx == m_king_vertex[color]);
        p = R_KING;
    }

    if (color == Types::BLACK) {
        p += 7;
    }

    return p;
}

void Board::generate_king_move(Types::Color color, std::vector<Move> &MoveList) const {

    const auto v = m_king_vertex[color];
    const auto mask = m_king_mask[v];
    const auto block_bitboard = mask & m_bb_color[color];
    auto legal_bitboard = mask ^ block_bitboard;

    while(legal_bitboard) {
        const auto res = BitUtils::lsb(legal_bitboard);
        assert(res != Types::NO_VERTEX);

        const auto from = v;
        const auto to = res;

        legal_bitboard = BitUtils::reset_ls1b(legal_bitboard);
        MoveList.emplace_back(std::move(Move(from, to)));
    }
}
