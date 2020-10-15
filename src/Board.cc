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
#include "Random.h"

#include <algorithm>
#include <sstream>
#include <functional>

constexpr std::array<Types::Piece, Board::NUM_VERTICES> Board::START_VERTICES;

constexpr std::array<Types::Direction, 8> Board::m_dirs;

std::array<std::array<int, Board::INTERSECTIONS>, Board::NUM_SYMMETRIES> Board::symmetry_nn_idx_table;
std::array<std::array<int, Board::NUM_VERTICES>, Board::NUM_SYMMETRIES> Board::symmetry_nn_vtx_table;

std::array<std::array<BitBoard, Board::NUM_VERTICES>, 2> Board::m_pawn_attacks;
std::array<Board::Magic, Board::NUM_VERTICES> Board::m_horse_magics;
std::array<Board::Magic, Board::NUM_VERTICES> Board::m_elephant_magics;
std::array<BitBoard, Board::NUM_VERTICES> Board::m_advisor_attacks;
std::array<BitBoard, Board::NUM_VERTICES> Board::m_king_attacks;

void Board::reset_board() {

    m_tomove = Types::RED;
    m_movenum = 0;

    auto start_position = get_start_position();
    fen2board(start_position);

    m_hash = calc_hash();
    m_lastmove = Move();
}

bool Board::fen2board(std::string &fen) {

    auto king_vertex_black = Types::NO_VERTEX;
    auto king_vertex_red = Types::NO_VERTEX;     

    auto bb_black = BitBoard(0ULL);
    auto bb_red = BitBoard(0ULL);

    auto bb_pawn = BitBoard(0ULL);
    auto bb_horse = BitBoard(0ULL);
    auto bb_rook = BitBoard(0ULL);
    auto bb_elephant = BitBoard(0ULL);
    auto bb_advisor = BitBoard(0ULL);
    auto bb_cannon = BitBoard(0ULL);


    auto fen_format = std::stringstream{fen};
    auto fen_stream = std::string{};

    fen_format >> fen_stream;

    auto success = bool{true};

    auto vtx = Types::VTX_A9;
    for (const auto &c : fen_stream) {
        bool skip = false;
        if (c == 'p') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_pawn |= bb;
            bb_black |= bb;
        } else if (c == 'c') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_cannon |= bb;
            bb_black |= bb;
        } else if (c == 'r') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_rook |= bb;
            bb_black |= bb;
        } else if (c == 'n') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_horse |= bb;
            bb_black |= bb;
        } else if (c == 'b') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_elephant |= bb;
            bb_black |= bb;
        } else if (c == 'a') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_advisor |= bb;
            bb_black |= bb;
        } else if (c == 'k') {
            king_vertex_black = vtx;
            bb_black |= Utils::vertex2bitboard(vtx);
        } else if (c == 'P') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_pawn |= bb;
            bb_red |= bb;
        } else if (c == 'C') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_cannon |= bb;
            bb_red|= bb;
        } else if (c == 'R') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_rook |= bb;
            bb_red |= bb;
        } else if (c == 'N') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_horse |= bb;
            bb_red |= bb;
        } else if (c == 'B') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_elephant |= bb;
            bb_red |= bb;
        } else if (c == 'A') {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_advisor |= bb;
            bb_red |= bb;
        } else if (c == 'K') {
            king_vertex_red = vtx;
            bb_red |= Utils::vertex2bitboard(vtx);
        } else if (c >= '1' && c <= '9') {
            vtx += (std::atoi(&c) - 1);
        } else if (c == '/') {
            if(is_on_board(vtx)) {
                success = false;
                break;
            }
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

    if (success) {
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

    return success;
}

std::pair<int, int> Board::get_symmetry(const int x,
                                        const int y,
                                        const int symmetry) {

    assert(x >= 0 || x < WIDTH);
    assert(y >= 0 || y < HEIGHT);
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    int idx_x = x;
    int idx_y = y;

    static constexpr auto REMAIN_WIDTH = WIDTH - 1;
    static constexpr auto REMAIN_HEIGHT = HEIGHT - 1;

    if ((symmetry & 2) != 0) {
        idx_x = REMAIN_WIDTH - idx_x;
    }

    if ((symmetry & 1) != 0) {
        idx_y = REMAIN_HEIGHT - idx_y;
    }

    assert(idx_x >= 0 && idx_x < WIDTH);
    assert(idx_y >= 0 && idx_y < HEIGHT);
    assert(symmetry != IDENTITY_SYMMETRY || (x == idx_x && y == idx_y));

    return {idx_x, idx_y};
}

void Board::init_symmetry() {

    for (auto &tables :  symmetry_nn_vtx_table) {
        for (auto &table : tables) {
            table = Types::NO_VERTEX;
        }
    }

    for (auto s = size_t{0}; s < NUM_SYMMETRIES; ++s) {
        for (int idx = 0; idx < INTERSECTIONS; ++idx) {
            const auto x = idx % WIDTH;
            const auto y = idx / WIDTH;
            const auto res = get_symmetry(x, y, s);
            symmetry_nn_idx_table[s][idx] = get_index(res.first, res.second);

            const auto vtx = get_vertex(x, y);
            symmetry_nn_vtx_table[s][vtx] = get_vertex(res.first, res.second);
        }
    }
}

void Board::init_pawn_attacks() {
    const auto lambda_pawn_attacks = [](const int vtx, const int color) -> BitBoard {

        auto BitBoard = tie(0ULL, 0ULL);
        if (color == Types::BLACK) {
            BitBoard |= Utils::vertex2bitboard(vtx + Types::SOUTH);
            if (Utils::on_area(vtx, RedSide)) {
                BitBoard |= Utils::vertex2bitboard(vtx + Types::WEST);
                BitBoard |= Utils::vertex2bitboard(vtx + Types::EAST);
            }
            BitBoard &= onBoard;
        } else if (color == Types::RED) {
            BitBoard |= Utils::vertex2bitboard(vtx + Types::NORTH);
            if (Utils::on_area(vtx, BlackSide)) {
                BitBoard |= Utils::vertex2bitboard(vtx + Types::WEST);
                BitBoard |= Utils::vertex2bitboard(vtx + Types::EAST);
            }
            BitBoard &= onBoard;
        }

        return BitBoard;
    };

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const auto vtx = get_vertex(x, y);
            m_pawn_attacks[Types::RED][vtx] = lambda_pawn_attacks(vtx, Types::RED);
            m_pawn_attacks[Types::BLACK][vtx] = lambda_pawn_attacks(vtx, Types::BLACK);
        }
    }
}

void Board::init_move_pattens() {

    init_pawn_attacks();

    // horse mask
    for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
        const auto bb = Utils::vertex2bitboard(v);
        auto mask = BitBoard(0ULL);
        if (Utils::on_board(bb)) {
            for (int k = 0; k < 4; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = Utils::shift(dir, bb);
                mask |= k_bb;
            }
        }
        m_horse_magics[v].mask = mask;
    }

    // elephant magic
    for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
        const auto bb = Utils::vertex2bitboard(v);
        auto mask = BitBoard(0ULL);
        if (Utils::on_board(bb)) {
            for (int k = 4; k < 8; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = Utils::shift(dir, bb);
                mask |= k_bb;
            }
        }
        m_elephant_magics[v].mask = mask;
    }

    // advisor attacks
    for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
        const auto bb = Utils::vertex2bitboard(v);
        auto mask = BitBoard(0ULL);
        if (Utils::on_area(bb, KingArea)) {
            for (int k = 4; k < 8; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = Utils::shift(dir, bb);
                mask |= k_bb;
            }
            mask &= KingArea;
        }
        m_advisor_attacks[v] = mask;
    }

    // king attacks
    for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
        const auto bb = Utils::vertex2bitboard(v);
        auto mask = BitBoard(0ULL);
        if (Utils::on_area(bb, KingArea)) {
            for (int k = 0; k < 4; ++k) {
                const auto dir = Board::m_dirs[k];
                const auto k_bb = Utils::shift(dir, bb);
                mask |= k_bb;
            }
            mask &= KingArea;
        }
        m_king_attacks[v] = mask;
    }
}

void Board::init_magics() {

    const auto set_valid = [](std::array<Magic, NUM_VERTICES> &magics) -> void {
        std::for_each(std::begin(magics), std::end(magics), [](auto &in){
                          if (in.mask != BitBoard(0ULL)) {
                              in.valid = true;
                          } else {
                              in.valid = false;
                          }
                      }); 
    };

    const auto generate_magic = [&](Types::Vertices v,
                                   std::array<Magic, NUM_VERTICES> &magics,
                                   std::function<BitBoard(BitBoard &, BitBoard &)> generate_reference) -> void {
        if (!magics[v].valid) {
            return;
        }
        auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
        auto center = Utils::vertex2bitboard(v);
        auto mask = magics[v].mask;
        const auto count = Utils::count(mask);
        const auto begin = 0ULL;
        const auto end = 1ULL << count;

        magics[v].shift = 64 - count;
        magics[v].attacks.resize(end);
        magics[v].attacks.shrink_to_fit();
        magics[v].limit = magics[v].attacks.size();

        auto vtxs = std::vector<Types::Vertices>{};
        while (mask) {
            const auto vtx = Utils::extract(mask);
            vtxs.emplace_back(vtx);
        }
        assert(vtxs.size() == (size_t)count);
        auto bit_iterator = Utils::BitIterator((size_t)count);
        for (auto b = begin; b < end; ++b) {
            if (b == begin) {
                bit_iterator.set(0ULL);
                auto zeroBB = BitBoard(0ULL);
                std::fill(std::begin(magics[v].attacks),
                          std::end(magics[v].attacks), zeroBB);
                magics[v].magic = rng.randuint64();
            }

            bit_iterator.next();
            auto res = bit_iterator.get();
            auto occupancy = BitBoard(0ULL);
            for (int i = 0; i < count; ++i) {
                if (res[i]) {
                    occupancy |= Utils::vertex2bitboard(vtxs[i]);
                }
            }

            auto reference = generate_reference(center, occupancy);

            if (!reference) {
                continue;
            }

            const auto index = magics[v].index(occupancy);
            if (magics[v].attacks[index]) {
                if (magics[v].attacks[index] != reference) {
                    b = begin-1;
                }
            } else {
                magics[v].attacks[index] = reference;
            }
        }
    };
  

    const auto elephant_reference = [&](BitBoard & center,
                                        BitBoard & occupancy) -> BitBoard {
        auto reference = BitBoard(0ULL);
        for (int k = 4; k < 8; ++k) {
            const auto dir = Board::m_dirs[k];
            const auto diagonal = Utils::shift(dir, center);
            if (!(diagonal & occupancy)) {
                reference |= Utils::shift(dir, diagonal);
            }
        }
        return reference;
    };

    const auto horse_reference = [&](BitBoard & center,
                                     BitBoard & occupancy) -> BitBoard {
        auto reference = BitBoard(0ULL);
        auto block = BitBoard(0ULL);

        block |= Utils::shift(Types::NORTH, center);
        block |= Utils::shift(Types::EAST, center);
        block |= Utils::shift(Types::SOUTH, center);
        block |= Utils::shift(Types::WEST, center);

        for (int k = 0; k < 4; ++k) {
            const auto dir = Board::m_dirs[k];
            const auto side = Utils::shift(dir, center);
            if (!(side & occupancy)) {
                for (int kk = 4; kk < 8; ++kk) {
                    const auto kdir = Board::m_dirs[kk];
                    const auto diagonal = Utils::shift(kdir, side);
                    if (!(diagonal & block)) {
                        reference |= diagonal;
                    }
                }
            }
        }   
        return reference;
    };

    // Utils::Timer timer;
    set_valid(m_elephant_magics);
    set_valid(m_horse_magics);

    for (auto vtx = Types::VTX_BEGIN; vtx < Types::VTX_END; ++vtx) {
        generate_magic(vtx, m_elephant_magics, elephant_reference);
        generate_magic(vtx, m_horse_magics, horse_reference);
    }
    // auto t = timer.get_duration();
    // printf("%f second(s)\n", t);
}

void Board::pre_initialize() {
    init_move_pattens();
    init_magics();
    init_symmetry();
}

void Board::piece_stream(std::ostream &out, Types::Piece p) const {

    p == Types::R_PAWN      ? out << "P" : p == Types::B_PAWN      ? out << "p" :
    p == Types::R_HORSE     ? out << "N" : p == Types::B_HORSE     ? out << "n" :
    p == Types::R_CANNON    ? out << "C" : p == Types::B_CANNON    ? out << "c" :
    p == Types::R_ROOK      ? out << "R" : p == Types::B_ROOK      ? out << "r" :
    p == Types::R_ELEPHANT  ? out << "B" : p == Types::B_ELEPHANT  ? out << "b" :
    p == Types::R_ADVISOR   ? out << "A" : p == Types::B_ADVISOR   ? out << "a" :
    p == Types::R_KING      ? out << "K" : p == Types::B_KING      ? out << "k" :
    p == Types::EMPTY_PIECE ? out << " " : out << "error";
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
            if (pis == Types::EMPTY_PIECE) { 
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

std::uint64_t Board::calc_hash(const int symmetry) const {

    auto res = Zobrist::zobrist_empty;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const auto svtx = symmetry_nn_vtx_table[symmetry][get_vertex(x, y)];
            const auto pis = get_piece(svtx);
            if (is_on_board(svtx)) {
                res ^= Zobrist::zobrist[pis][svtx];
            } 
        }
    }
    if (m_tomove == Types::RED) {
        res ^= Zobrist::zobrist_redtomove;
    }
    return res;
}

bool Board::is_on_board(const int vtx) const {
    return START_VERTICES[vtx] != Types::INVAL_PIECE;
}

void Board::dump_board() const {
    auto out = std::ostringstream{};
    board_stream(out);
    Utils::auto_printf(out);
}

std::string Board::get_start_position() {
    return std::string{"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"};
}


Types::Piece Board::get_piece(const int x, const int y) const {
    return get_piece(get_vertex(x, y));
}


Types::Piece Board::get_piece(const int vtx) const {

    auto color = Types::INVALID_COLOR;
    auto p = Types::INVAL_PIECE;

    if (Utils::on_area(vtx, m_bb_color[Types::RED])) {
        color = Types::RED;
    } else if (Utils::on_area(vtx, m_bb_color[Types::BLACK])) {
        color = Types::BLACK;
    }

    if (color == Types::INVALID_COLOR) {
        if (is_on_board(vtx)) {
            return Types::EMPTY_PIECE;
        } else {
            return Types::INVAL_PIECE;
        }
    }

    if (Utils::on_area(vtx, m_bb_pawn)) {
        p = Types::R_PAWN;
    } else if (Utils::on_area(vtx, m_bb_horse)) {
        p = Types::R_HORSE;
    } else if (Utils::on_area(vtx, m_bb_cannon)) {
        p = Types::R_CANNON;
    } else if (Utils::on_area(vtx, m_bb_rook)) {
        p = Types::R_ROOK;
    } else if (Utils::on_area(vtx, m_bb_elephant)) {
        p = Types::R_ELEPHANT;
    } else if (Utils::on_area(vtx, m_bb_advisor)) {
        p = Types::R_ADVISOR;
    } else {
        assert(vtx == m_king_vertex[color]);
        p = Types::R_KING;
    }

    if (color == Types::BLACK) {
        p += 7;
    }

    return p;
}

Types::Color Board::get_to_move() const {
    return m_tomove;
}

const auto lambda_separate_bitboarad = [](Types::Vertices vtx,
                                          BitBoard &legal_bitboard,
                                          std::vector<Move> &MoveList) -> void {
    while (legal_bitboard) {
        const auto res = Utils::extract(legal_bitboard);
        assert(res != Types::NO_VERTEX);

        const auto from = vtx;
        const auto to = res;
        MoveList.emplace_back(std::move(Move(from, to)));
    }
};


template<>
void Board::generate_move<Types::KING>(Types::Color color, std::vector<Move> &MoveList) const {

    const auto vtx = m_king_vertex[color];
    const auto attack = m_king_attacks[vtx];
    const auto block = attack & m_bb_color[color];
    auto legal_bitboard = attack ^ block;

    lambda_separate_bitboarad(vtx, legal_bitboard, MoveList);
}

template<>
void Board::generate_move<Types::PAWN>(Types::Color color, std::vector<Move> &MoveList) const {

    auto bb_p = m_bb_pawn & m_bb_color[color];
    while (bb_p) {
        const auto vtx = Utils::extract(bb_p);
        const auto attack = m_pawn_attacks[color][vtx];
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, MoveList);
    }
}

template<>
void Board::generate_move<Types::ADVISOR>(Types::Color color, std::vector<Move> &MoveList) const {

    auto bb_a = m_bb_advisor & m_bb_color[color];
    while (bb_a) {
        const auto vtx = Utils::extract(bb_a);
        const auto attack = m_advisor_attacks[vtx];
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, MoveList);
    }
}

template<>
void Board::generate_move<Types::ELEPHANT>(Types::Color color, std::vector<Move> &MoveList) const {

    auto bb_e = m_bb_elephant & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    while (bb_e) {
        const auto vtx = Utils::extract(bb_e);
        const auto attack = m_elephant_magics[vtx].attack(occupancy);
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, MoveList);
    }
}

template<>
void Board::generate_move<Types::HORSE>(Types::Color color, std::vector<Move> &MoveList) const {

    auto bb_h = m_bb_horse & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    while (bb_h) {
        const auto vtx = Utils::extract(bb_h);
        const auto attack = m_horse_magics[vtx].attack(occupancy);
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, MoveList);
    }
}


void Board::generate_movelist(Types::Color color, std::vector<Move> &MoveList) const {

    MoveList.clear();
    MoveList.reserve(option<int>("reserve_movelist"));

    generate_move<Types::KING>    (color, MoveList);
    generate_move<Types::PAWN>    (color, MoveList);
    generate_move<Types::HORSE>   (color, MoveList);
    generate_move<Types::ADVISOR> (color, MoveList);
    generate_move<Types::ELEPHANT>(color, MoveList);

    MoveList.shrink_to_fit();
}
