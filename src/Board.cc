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

#include "Board.h"
#include "Random.h"

#include <functional>
#include <algorithm>
#include <iterator>
#include <sstream>

constexpr std::array<Types::Piece, Board::NUM_VERTICES> Board::START_VERTICES;

constexpr std::array<Types::Direction, 8> Board::m_dirs;

std::array<std::array<int, Board::INTERSECTIONS>, Board::NUM_SYMMETRIES> Board::symmetry_nn_idx_table;
std::array<std::array<Types::Vertices, Board::NUM_VERTICES>, Board::NUM_SYMMETRIES> Board::symmetry_nn_vtx_table;

std::array<std::array<BitBoard, Board::NUM_VERTICES>, 2> Board::m_pawn_attacks;
std::array<BitBoard, Board::NUM_VERTICES> Board::m_advisor_attacks;
std::array<BitBoard, Board::NUM_VERTICES> Board::m_king_attacks;

std::array<Board::Magic, Board::NUM_VERTICES> Board::m_horse_magics;
std::array<Board::Magic, Board::NUM_VERTICES> Board::m_elephant_magics;

std::array<Board::Magic, Board::NUM_VERTICES> Board::m_rookrank_magics;
std::array<Board::Magic, Board::NUM_VERTICES> Board::m_rookfile_magics;

std::array<Board::Magic, Board::NUM_VERTICES> Board::m_cannonrank_magics;
std::array<Board::Magic, Board::NUM_VERTICES> Board::m_cannonfile_magics;

void Board::reset_board() {
    clear_status();
    auto start_position = get_start_position();
    fen2board(start_position);
    m_hash = calc_hash();
}

void Board::clear_status() {
    m_tomove = Types::RED;
    m_gameply = 1;
    m_movenum = 0;
    m_eaten = false;
    m_lastmove = Move{};
}

bool Board::fen2board(std::string &fen) {

    // FEN example:
    // rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
    //
    // part 1 : position
    // rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
    //
    // part 2 : to move
    // w
    //
    // part 3 : invalid
    // -
    //
    // part 4 : invalid
    // -
    //
    // part 5 : invalid
    // 0
    //
    // part 6 : move number
    // 1

    auto cnt_stream = std::stringstream{fen};
    const auto cnt = std::distance(std::istream_iterator<std::string>(cnt_stream),
                                   std::istream_iterator<std::string>());

    if (cnt != 6) {
        return false;
    }

    auto fen_format = std::stringstream{fen};
    auto fen_stream = std::string{};

    auto success = bool{true};
    auto bb_black = BitBoard(0ULL);
    auto bb_red = BitBoard(0ULL);

    auto king_vertex_black = Types::NO_VERTEX;
    auto king_vertex_red = Types::NO_VERTEX;
    auto bb_pawn = BitBoard(0ULL);
    auto bb_horse = BitBoard(0ULL);
    auto bb_rook = BitBoard(0ULL);
    auto bb_elephant = BitBoard(0ULL);
    auto bb_advisor = BitBoard(0ULL);
    auto bb_cannon = BitBoard(0ULL);

    // part 1 : position
    fen_format >> fen_stream;
    const auto black_pawn_en = option<char>("black_pawn_en");
    const auto black_cannon_en = option<char>("black_cannon_en");
    const auto black_rook_en = option<char>("black_rook_en");
    const auto black_horse_en = option<char>("black_horse_en");
    const auto black_elephant_en = option<char>("black_elephant_en");
    const auto black_advisor_en = option<char>("black_advisor_en");
    const auto black_king_en = option<char>("black_king_en");

    const auto red_pawn_en = option<char>("red_pawn_en");
    const auto red_cannon_en = option<char>("red_cannon_en");
    const auto red_rook_en = option<char>("red_rook_en");
    const auto red_horse_en = option<char>("red_horse_en");
    const auto red_elephant_en = option<char>("red_elephant_en");
    const auto red_advisor_en = option<char>("red_advisor_en");
    const auto red_king_en = option<char>("red_king_en");

    auto vtx = Types::VTX_A9;
    for (const char &c : fen_stream) {
        bool skip = false;
        if (c == black_pawn_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_pawn |= bb;
            bb_black |= bb;
        } else if (c == black_cannon_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_cannon |= bb;
            bb_black |= bb;
        } else if (c == black_rook_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_rook |= bb;
            bb_black |= bb;
        } else if (c == black_horse_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_horse |= bb;
            bb_black |= bb;
        } else if (c == black_elephant_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_elephant |= bb;
            bb_black |= bb;
        } else if (c == black_advisor_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_advisor |= bb;
            bb_black |= bb;
        } else if (c == black_king_en) {
            king_vertex_black = vtx;
            bb_black |= Utils::vertex2bitboard(vtx);
        } else if (c == red_pawn_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_pawn |= bb;
            bb_red |= bb;
        } else if (c == red_cannon_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_cannon |= bb;
            bb_red|= bb;
        } else if (c == red_rook_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_rook |= bb;
            bb_red |= bb;
        } else if (c == red_horse_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_horse |= bb;
            bb_red |= bb;
        } else if (c == red_elephant_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_elephant |= bb;
            bb_red |= bb;
        } else if (c == red_advisor_en) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_advisor |= bb;
            bb_red |= bb;
        } else if (c == red_king_en) {
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
        } else {
            success = false;
            break;
        }

        if (vtx == Types::VTX_J0) {
            break;
        }
        if (!skip) {
            ++vtx;
        }
    }

    // part 2 : to move
    fen_format >> fen_stream;
    auto tomove = Types::INVALID_COLOR;
    if (fen_stream == "w" || fen_stream == "r") {
        tomove = Types::RED;
    } else if (fen_stream == "b") {
        tomove = Types::BLACK;
    } else {
        success = false;
    }

    // part 3-5 : invalid
    for (int k = 0; k < 3; ++k) {
        fen_format >> fen_stream;
    }

    // part 6 : turn number
    int ply;
    fen_format >> ply;
    if (ply <= 0) {
        success = false;
    }

    if (success) {
        clear_status();
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
        m_gameply = ply;
        m_movenum = (ply-1) * 2 + static_cast<int>(tomove);
        m_tomove = tomove;

        // Calculate the new hash value. 
        m_hash = calc_hash();
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

    static constexpr auto MIRROR_WIDTH = WIDTH - 1;
    static constexpr auto MIRROR_HEIGHT = HEIGHT - 1;

    if ((symmetry & 2) != 0) {
        idx_x = MIRROR_WIDTH - idx_x;
    }

    if ((symmetry & 1) != 0) {
        idx_y = MIRROR_HEIGHT - idx_y;
    }

    assert(idx_x >= 0 && idx_x < WIDTH);
    assert(idx_y >= 0 && idx_y < HEIGHT);
    assert(symmetry != IDENTITY_SYMMETRY || (x == idx_x && y == idx_y));

    return {idx_x, idx_y};
}

void Board::init_symmetry() {

    for (auto &tables : symmetry_nn_vtx_table) {
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

// Initialize attack table and magic mask.
void Board::init_move_pattens() {

    init_pawn_attacks();

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

    // horse magics
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

    // elephant magics
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

    // rank and file magics
    for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
        const auto x = get_x(v);
        const auto y = get_y(v);
        const auto rankmask = (Rank0BB << (BITBOARD_SHIFT * y)) & onBoard;
        auto filemask = (FileABB << x) & onBoard;

        m_rookrank_magics[v].mask = rankmask;
        m_rookfile_magics[v].mask = filemask;
        m_cannonrank_magics[v].mask = rankmask;
        m_cannonfile_magics[v].mask = filemask;

        const auto test = rankmask & filemask;
        if (test) {
            assert(test == Utils::vertex2bitboard(v));
        }
    }
}

// Initialize the magic numbers.
void Board::init_magics() {

    // Find valid bitboard.
    const auto set_valid = [](std::array<Magic, NUM_VERTICES> &magics) -> void {
        std::for_each(std::begin(magics), std::end(magics), [](auto &in){
                          if (in.mask != BitBoard(0ULL)) {
                              in.valid = true;
                          } else {
                              in.valid = false;
                          }
                      }); 
    };

    // Generate the magic numbers.
    const auto generate_magics = [&](const int addition, Types::Vertices v,
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
        const auto attacksize = 1ULL << (count + addition);

        magics[v].shift = 64 - (count + addition);
        magics[v].attacks.resize(attacksize);
        magics[v].attacks.shrink_to_fit();
        magics[v].limit = magics[v].attacks.size();
        auto used = std::vector<bool>(attacksize);

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
                std::fill(std::begin(used), std::end(used), false);
                magics[v].upper_magic = rng.randuint64();
                magics[v].lower_magic = rng.randuint64();
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
            if (!used[index]) {
                magics[v].attacks[index] = reference;
                used[index] = true;
            } else {
                if (magics[v].attacks[index] != reference) {
                    b = begin-1;
                }
            }
        }
    };

    const auto elephant_reference = [&](BitBoard &center,
                                        BitBoard &occupancy) -> BitBoard {
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

    const auto horse_reference = [&](BitBoard &center,
                                     BitBoard &occupancy) -> BitBoard {
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

    const auto rookrank_reference = [&](BitBoard &center,
                                        BitBoard &occupancy) -> BitBoard {
        auto reference = BitBoard(0ULL);
        auto dirs = {Types::EAST, Types::WEST}; 
        for (const auto &dir : dirs) {
            auto p = center;
            do {
                p = Utils::shift(dir, p);
                reference |= p;
            } while (Utils::on_board(p) && !(p & occupancy));
        }
        return reference;
    };

    const auto rookfile_reference = [&](BitBoard &center,
                                        BitBoard &occupancy) -> BitBoard {
        auto reference = BitBoard(0ULL);
        auto dirs = {Types::NORTH, Types::SOUTH}; 
        for (const auto &dir : dirs) {
            auto p = center;
            do {
                p = Utils::shift(dir, p);
                reference |= p;
            } while (Utils::on_board(p) && !(p & occupancy));
        }
        return reference;
    };

    const auto cannonrank_reference = [&](BitBoard &center,
                                        BitBoard &occupancy) -> BitBoard {
        auto reference = BitBoard(0ULL);
        auto dirs = {Types::EAST, Types::WEST}; 
        for (const auto &dir : dirs) {
            auto p = Utils::shift(dir, center);
            while (Utils::on_board(p) && !(p & occupancy)) {
                reference |= p;
                p = Utils::shift(dir, p);
            }

            do {
                p = Utils::shift(dir, p);
            } while (Utils::on_board(p) && !(p & occupancy));

            reference |= p;
        }
        return reference;
    };

    const auto cannonfile_reference = [&](BitBoard &center,
                                          BitBoard &occupancy) -> BitBoard {
        auto reference = BitBoard(0ULL);
        auto dirs = {Types::NORTH, Types::SOUTH}; 
        for (const auto &dir : dirs) {
            auto p = Utils::shift(dir, center);
            while (Utils::on_board(p) && !(p & occupancy)) {
                reference |= p;
                p = Utils::shift(dir, p);
            }

            do {
                p = Utils::shift(dir, p);
            } while (Utils::on_board(p) && !(p & occupancy));

            reference |= p;
        }
        return reference;
    };

    Utils::Timer timer;
    set_valid(m_elephant_magics);
    set_valid(m_horse_magics);
    set_valid(m_rookrank_magics);
    set_valid(m_rookfile_magics);
    set_valid(m_cannonrank_magics);
    set_valid(m_cannonfile_magics);

    for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
        generate_magics(0, v, m_elephant_magics, elephant_reference);
        generate_magics(0, v, m_horse_magics, horse_reference);
        generate_magics(2, v, m_rookrank_magics, rookrank_reference);
        generate_magics(4, v, m_rookfile_magics, rookfile_reference);
        generate_magics(2, v, m_cannonrank_magics, cannonrank_reference);
        generate_magics(4, v, m_cannonfile_magics, cannonfile_reference);
    }

    auto t = timer.get_duration();
    Utils::printf<Utils::STATS>("Generating Magic numbers spent %.4f second(s)\n", t);
}

void Board::dump_memory() {

    auto res = size_t{0};
    res += sizeof(m_pawn_attacks);
    res += sizeof(m_advisor_attacks);
    res += sizeof(m_king_attacks);

    res += sizeof(m_horse_magics);
    res += sizeof(m_elephant_magics);
    res += sizeof(m_rookrank_magics);
    res += sizeof(m_rookfile_magics);
    res += sizeof(m_cannonrank_magics);
    res += sizeof(m_cannonfile_magics);

    for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
        res += sizeof(BitBoard) * m_horse_magics[v].attacks.capacity();
        res += sizeof(BitBoard) * m_elephant_magics[v].attacks.capacity();
        res += sizeof(BitBoard) * m_rookrank_magics[v].attacks.capacity();
        res += sizeof(BitBoard) * m_rookfile_magics[v].attacks.capacity();
        res += sizeof(BitBoard) * m_cannonrank_magics[v].attacks.capacity();
        res += sizeof(BitBoard) * m_cannonfile_magics[v].attacks.capacity();
    }
    Utils::printf<Utils::STATS>("Attacks Table Memory : %.4f (Mib)\n", (double)res / (1024.f * 1024.f));
}

void Board::pre_initialize() {
    init_move_pattens();
    init_magics();
    init_symmetry();
    dump_memory();
}

template<>
void Board::piece_stream<Types::ASCII>(std::ostream &out, Types::Piece p) {
    p == Types::R_PAWN      ? out << option<char>("red_pawn_en")     : p == Types::B_PAWN      ? out << option<char>("black_pawn_en")     :
    p == Types::R_HORSE     ? out << option<char>("red_horse_en")    : p == Types::B_HORSE     ? out << option<char>("black_horse_en")    :
    p == Types::R_CANNON    ? out << option<char>("red_cannon_en")   : p == Types::B_CANNON    ? out << option<char>("black_cannon_en")   :
    p == Types::R_ROOK      ? out << option<char>("red_rook_en")     : p == Types::B_ROOK      ? out << option<char>("black_rook_en")     :
    p == Types::R_ELEPHANT  ? out << option<char>("red_elephant_en") : p == Types::B_ELEPHANT  ? out << option<char>("black_elephant_en") :
    p == Types::R_ADVISOR   ? out << option<char>("red_advisor_en")  : p == Types::B_ADVISOR   ? out << option<char>("black_advisor_en")  :
    p == Types::R_KING      ? out << option<char>("red_king_en")     : p == Types::B_KING      ? out << option<char>("black_king_en")     :
    p == Types::EMPTY_PIECE ? out << " " : out << "error";
}

template<>
void Board::piece_stream<Types::TRADITIONAL_CHINESE>(std::ostream &out, Types::Piece p) {
    using STR = std::string;
    p == Types::R_PAWN      ? out << option<STR>("red_pawn_ch")     : p == Types::B_PAWN      ? out << option<STR>("black_pawn_ch")     :
    p == Types::R_HORSE     ? out << option<STR>("red_horse_ch")    : p == Types::B_HORSE     ? out << option<STR>("black_horse_ch")    :
    p == Types::R_CANNON    ? out << option<STR>("red_cannon_ch")   : p == Types::B_CANNON    ? out << option<STR>("black_cannon_ch")   :
    p == Types::R_ROOK      ? out << option<STR>("red_rook_ch")     : p == Types::B_ROOK      ? out << option<STR>("black_rook_ch")     :
    p == Types::R_ELEPHANT  ? out << option<STR>("red_elephant_ch") : p == Types::B_ELEPHANT  ? out << option<STR>("black_elephant_ch") :
    p == Types::R_ADVISOR   ? out << option<STR>("red_advisor_ch")  : p == Types::B_ADVISOR   ? out << option<STR>("black_advisor_ch")  :
    p == Types::R_KING      ? out << option<STR>("red_king_ch")     : p == Types::B_KING      ? out << option<STR>("black_king_ch")     :
    p == Types::EMPTY_PIECE ? out << "  " : out << "error";
}

template<>
void Board::info_stream<Types::ASCII>(std::ostream &out) const {

    out << "{";
    if (m_tomove == Types::RED) {
        out << "Next player : RED";
    } else if (m_tomove == Types::BLACK) {
        out << "Next player : BLACK";
    }  else {
        out << "color error!";
    }

    out << ", Last move : ";
    out << get_last_move().to_string(); 

    out << ", Move number : ";
    out << get_movenum(); 

    out << ", Hash : ";
    out << std::hex;
    out << get_hash();
    out << std::dec;

    out << ",\n Fen : ";
    fen_stream(out);
    out << "}";
    out << std::endl;
}

template<>
void Board::info_stream<Types::TRADITIONAL_CHINESE>(std::ostream &out) const {

    out << "{";
    if (m_tomove == Types::RED) {
        out << "下一手 ：紅方";
    } else if (m_tomove == Types::BLACK) {
        out << "下一手 ：黑方";
    }  else {
        out << "color error!";
    }

    out << "，上一手棋 ：";
    out << get_last_move().to_string(); 

    out << "，第 ";
    out << get_movenum(); 
    out << " 手棋";

    out << "，哈希 ：";
    out << std::hex;
    out << get_hash();
    out << std::dec;

    out << "，\n Fen ：";
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
            piece_stream<Types::ASCII>(out, pis);
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

    out << " - - 0 " << m_gameply;
}

template<>
void Board::board_stream<Types::ASCII>(std::ostream &out, const Move lastmove) const {
    const auto from_vertex = lastmove.get_from();
    const auto to_vertex = lastmove.get_to();
    for (int y = 0; y < HEIGHT; ++y) {
        Utils::space_stream(out, 1);
        out << "+---+---+---+---+---+---+---+---+---+";
        Utils::strip_stream(out, 1);
        auto mark = false;
        for (int x = 0; x < WIDTH; ++x) {
            const auto coordinate_x = x;
            const auto coordinate_y = HEIGHT - y - 1;
            const auto coordinate_vtx = get_vertex(coordinate_x, coordinate_y);
            if (mark) {
                mark = false;
                out << ")";
            } else {
                out << " ";
            }

            out << "|";

            if (coordinate_vtx == to_vertex) {
                mark = true;
                out << "(";
            } else {
                out << " ";
            }
            if (coordinate_vtx == from_vertex) {
                out << "*";
            } else {
                piece_stream<Types::ASCII>(out, coordinate_x, coordinate_y);
            }
        }
        if (mark) {
            mark = false;
            out << ")";
        } else {
            out << " ";
        }
        out << "| ";
        out << HEIGHT - y - 1;
        Utils::strip_stream(out, 1);
    }
    Utils::space_stream(out, 1);
    out << "+---+---+---+---+---+---+---+---+---+" << std::endl;
    out << "   a   b   c   d   e   f   g   h   i";
    Utils::strip_stream(out, 1);

    info_stream<Types::ASCII>(out);
}

template<>
void Board::board_stream<Types::TRADITIONAL_CHINESE>(std::ostream &out, const Move lastmove) const {
    const auto to_vertex = lastmove.get_to();
    for (int y = 0; y < HEIGHT; ++y) {
        Utils::space_stream(out, 1);
        out << "+----+----+----+----+----+----+----+----+----+";
        Utils::strip_stream(out, 1);
        auto mark = false;
        for (int x = 0; x < WIDTH; ++x) {
            const auto coordinate_x = x;
            const auto coordinate_y = HEIGHT - y - 1;
            const auto coordinate_vtx = get_vertex(coordinate_x, coordinate_y);
            if (mark) {
                mark = false;
                out << ")";
            } else {
                out << " ";
            }

            out << "|";

            if (coordinate_vtx == to_vertex) {
                mark = true;
                out << "(";
            } else {
                out << " ";
            }
            piece_stream<Types::TRADITIONAL_CHINESE>(out, coordinate_x, coordinate_y);
        }
        if (mark) {
            mark = false;
            out << ")";
        } else {
            out << " ";
        }
        out << "| ";
        out << HEIGHT - y - 1;
        Utils::strip_stream(out, 1);
    }
    Utils::space_stream(out, 1);
    out << "+----+----+----+----+----+----+----+----+----+" << std::endl;
    out << "    a    b    c    d    e    f    g    h    i";
    Utils::strip_stream(out, 2);

    info_stream<Types::TRADITIONAL_CHINESE>(out);
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

bool Board::is_on_board(const Types::Vertices vtx) {
    return START_VERTICES[vtx] != Types::INVAL_PIECE;
}

std::string Board::get_start_position() {
    return std::string{"rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"};
}


Types::Piece Board::get_piece(const int x, const int y) const {
    return get_piece(get_vertex(x, y));
}

Types::Piece Board::get_piece(const Types::Vertices vtx) const {

    auto color = Types::INVALID_COLOR;

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

    auto p = static_cast<Types::Piece>(get_piece_type(vtx));

    if (color == Types::BLACK) {
        p += 7;
    }

    return p;
}

Types::Piece_t Board::get_piece_type(const Types::Vertices vtx) const {

    auto pt = Types::EMPTY_PIECE_T;
    auto at = Utils::vertex2bitboard(vtx);

    if (at & m_bb_pawn) {
        pt = Types::PAWN;
    } else if (at & m_bb_horse) {
        pt = Types::HORSE;
    } else if (at & m_bb_cannon) {
        pt = Types::CANNON;
    } else if (at & m_bb_rook) {
        pt = Types::ROOK;
    } else if (at & m_bb_elephant) {
        pt = Types::ELEPHANT;
    } else if (at & m_bb_advisor) {
        pt = Types::ADVISOR;
    } else if (vtx == m_king_vertex[Types::RED] ||
               vtx == m_king_vertex[Types::BLACK]) {
        pt = Types::KING;
    }

    return pt;
}

BitBoard &Board::get_piece_bitboard_ref(Types::Piece_t pt) {

    assert(pt != Types::KING);
    if (pt == Types::HORSE) {
        return m_bb_horse;
    } else if (pt == Types::ROOK) {
        return m_bb_rook;
    } else if (pt == Types::ELEPHANT) {
        return m_bb_elephant;
    } else if (pt == Types::ADVISOR) {
        return m_bb_advisor;
    } else if (pt == Types::CANNON) {
        return m_bb_cannon;
    }

    assert(pt == Types::PAWN);
    return m_bb_pawn;
}

const auto lambda_separate_bitboarad = [](Types::Vertices vtx,
                                          BitBoard &legal_bitboard,
                                          std::vector<Move> &movelist) -> int {
    int cnt = 0;
    while (legal_bitboard) {
        ++cnt;
        const auto res = Utils::extract(legal_bitboard);
        assert(res != Types::NO_VERTEX);

        const auto from = vtx;
        const auto to = res;
        movelist.emplace_back(Move(from, to));
    }
    return cnt;
};

template<>
int Board::generate_move<Types::KING>(Types::Color color, std::vector<Move> &movelist) const {

    const auto vtx = m_king_vertex[color];
    const auto attack = m_king_attacks[vtx];
    const auto block = attack & m_bb_color[color];
    auto legal_bitboard = attack ^ block;

    auto cnt = lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
    return cnt;
}

template<>
int Board::generate_move<Types::PAWN>(Types::Color color, std::vector<Move> &movelist) const {
    int cnt = 0;
    auto bb_p = m_bb_pawn & m_bb_color[color];
    while (bb_p) {
        const auto vtx = Utils::extract(bb_p);
        const auto attack = m_pawn_attacks[color][vtx];
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        cnt += lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
    }
    return cnt;
}

template<>
int Board::generate_move<Types::ADVISOR>(Types::Color color, std::vector<Move> &movelist) const {
    int cnt = 0;
    auto bb_a = m_bb_advisor & m_bb_color[color];
    while (bb_a) {
        const auto vtx = Utils::extract(bb_a);
        const auto attack = m_advisor_attacks[vtx];
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        cnt += lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
    }
    return cnt;
}

template<>
int Board::generate_move<Types::ELEPHANT>(Types::Color color, std::vector<Move> &movelist) const {
    int cnt = 0;
    auto bb_e = m_bb_elephant & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    auto mask = color == Types::RED ? RedSide : BlackSide;
    while (bb_e) {
        const auto vtx = Utils::extract(bb_e);
        const auto attack = m_elephant_magics[vtx].attack(occupancy);
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = (attack ^ block) & mask;
        cnt += lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
    }
    return cnt;
}

template<>
int Board::generate_move<Types::HORSE>(Types::Color color, std::vector<Move> &movelist) const {
    int cnt = 0;
    auto bb_h = m_bb_horse & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    while (bb_h) {
        const auto vtx = Utils::extract(bb_h);
        const auto attack = m_horse_magics[vtx].attack(occupancy);
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        cnt += lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
    }
    return cnt;
}

template<>
int Board::generate_move<Types::ROOK>(Types::Color color, std::vector<Move> &movelist) const {
    int cnt = 0;
    auto bb_r = m_bb_rook & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    while (bb_r) {
        const auto vtx = Utils::extract(bb_r);
        const auto rankattack = m_rookrank_magics[vtx].attack(occupancy);
        const auto fileattack = m_rookfile_magics[vtx].attack(occupancy);
        const auto attack = rankattack | fileattack;
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        cnt += lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
    }
    return cnt;
}

template<>
int Board::generate_move<Types::CANNON>(Types::Color color, std::vector<Move> &movelist) const {
    int cnt = 0;
    auto bb_c = m_bb_cannon & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    while (bb_c) {
        const auto vtx = Utils::extract(bb_c);
        const auto rankattack = m_cannonrank_magics[vtx].attack(occupancy);
        const auto fileattack = m_cannonfile_magics[vtx].attack(occupancy);
        const auto attack = rankattack | fileattack;
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        cnt += lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
    }
    return cnt;
}

// Generating the all pseudo legal moves to the list.
int Board::generate_pseudo_movelist(Types::Color color, std::vector<Move> &movelist) const {

    const auto reserve = option<int>("reserve_movelist");
    movelist.clear();
    movelist.reserve(reserve);

    int cnt = 0;

    cnt += generate_move<Types::KING>    (color, movelist);
    cnt += generate_move<Types::PAWN>    (color, movelist);
    cnt += generate_move<Types::ROOK>    (color, movelist);
    cnt += generate_move<Types::HORSE>   (color, movelist);
    cnt += generate_move<Types::CANNON>  (color, movelist);
    cnt += generate_move<Types::ADVISOR> (color, movelist);
    cnt += generate_move<Types::ELEPHANT>(color, movelist);

    if (cnt > reserve) {
        movelist.shrink_to_fit();
    }

    return cnt;
}

// Generating the all legal moves to the list.
int Board::generate_movelist(Types::Color color, std::vector<Move> &movelist) {
    auto cnt = generate_pseudo_movelist(color, movelist);
    movelist.erase(
        std::remove_if(std::begin(movelist), std::end(movelist),
                       [&](const auto &move) {
                           auto record = do_pseudo_move(move);
                           auto legal = true;
                           legal &= !is_king_face_king();
                           undo_from_pseudo_move(record);
                           return !legal;
                       }),
        std::end(movelist)
    );
    return cnt;
}

void Board::set_to_move(Types::Color color) {
    update_zobrist_tomove(color, m_tomove);
    m_tomove = color;
}

void Board::swap_to_move() {
    set_to_move(swap_color(m_tomove));
}

// Assume the move is legal.
void Board::do_move(Move move) {
    const auto from = move.get_from();
    const auto to = move.get_to();
    const auto form_bitboard =  move.get_from_bitboard();
    const auto to_bitboard = move.get_to_bitboard();

    // Get the color
    auto color = Types::INVALID_COLOR;
    if (m_bb_color[Types::BLACK] & form_bitboard) {
        color = Types::BLACK;
    } else if (m_bb_color[Types::RED] & form_bitboard) {
        color = Types::RED;
    }
    assert(color == m_tomove);

    // Get the piece type
    const auto pt = get_piece_type(from);
    assert(pt != Types::EMPTY_PIECE_T);

    // Eat piece
    const auto opp_color = swap_color(color);
    m_eaten = m_bb_color[opp_color] & to_bitboard;

    auto eaten_pt = Types::EMPTY_PIECE_T;
    if (m_eaten) {
        eaten_pt = get_piece_type(to);
        if (eaten_pt == Types::KING) {
            m_king_vertex[opp_color] = Types::NO_VERTEX;
        } else {
            auto &ref_bb = get_piece_bitboard_ref(eaten_pt);
            ref_bb ^= to_bitboard;
        }
        m_bb_color[opp_color] ^= to_bitboard;
    }

    // Update bitboard
    if (pt == Types::KING) {
        assert(m_king_vertex[color] == from);
        m_king_vertex[color] = to;
    } else {
        auto &ref_bb = get_piece_bitboard_ref(pt);
        ref_bb ^= form_bitboard;
        ref_bb ^= to_bitboard;
    }

    m_bb_color[color] ^= form_bitboard;
    m_bb_color[color] ^= to_bitboard;

    auto p = static_cast<Types::Piece>(pt) + (color == Types::BLACK ? 7 : 0);

    // Update last move
    m_lastmove = move;

    // Update zobrist
    update_zobrist(p , from, to);
    if (m_eaten) {
        auto eaten_p = static_cast<Types::Piece>(eaten_pt) + (color == Types::BLACK ? 7 : 0);
        update_zobrist_remove(eaten_p, to);
    } 

    // Swap color
    swap_to_move();

    // Increment move number
    increment_movenum();
}

Board::PseudoMoveRecord Board::do_pseudo_move(Move move) {
    auto record = PseudoMoveRecord{};
    record.eaten = m_eaten;
    record.bb_pawn = m_bb_pawn;
    record.bb_horse = m_bb_horse;
    record.bb_rook = m_bb_rook;
    record.bb_elephant = m_bb_elephant;
    record.bb_advisor = m_bb_advisor;
    record.bb_cannon = m_bb_cannon;
    record.king_vertex = m_king_vertex;
    record.bb_color = m_bb_color;

    const auto from = move.get_from();
    const auto to = move.get_to();
    const auto form_bitboard =  move.get_from_bitboard();
    const auto to_bitboard = move.get_to_bitboard();

    // Get the color
    auto color = Types::INVALID_COLOR;
    if (m_bb_color[Types::BLACK] & form_bitboard) {
        color = Types::BLACK;
    } else if (m_bb_color[Types::RED] & form_bitboard) {
        color = Types::RED;
    }
    assert(color == m_tomove);

    // Get the piece type
    const auto pt = get_piece_type(from);
    assert(pt != Types::EMPTY_PIECE_T);

    // Eat piece
    const auto opp_color = swap_color(color);
    m_eaten = m_bb_color[opp_color] & to_bitboard;

    auto eaten_pt = Types::EMPTY_PIECE_T;
    if (m_eaten) {
        eaten_pt = get_piece_type(to);
        if (eaten_pt == Types::KING) {
            m_king_vertex[opp_color] = Types::NO_VERTEX;
        } else {
            auto &ref_bb = get_piece_bitboard_ref(eaten_pt);
            ref_bb ^= to_bitboard;
        }
        m_bb_color[opp_color] ^= to_bitboard;
    }

    // Update bitboard
    if (pt == Types::KING) {
        assert(m_king_vertex[color] == from);
        m_king_vertex[color] = to;
    } else {
        auto &ref_bb = get_piece_bitboard_ref(pt);
        ref_bb ^= form_bitboard;
        ref_bb ^= to_bitboard;
    }

    m_bb_color[color] ^= form_bitboard;
    m_bb_color[color] ^= to_bitboard;

    return record;
}

void Board::undo_from_pseudo_move(Board::PseudoMoveRecord record) {
    m_eaten = record.eaten;
    m_bb_pawn = record.bb_pawn;
    m_bb_horse = record.bb_horse;
    m_bb_rook = record.bb_rook;
    m_bb_elephant = record.bb_elephant;
    m_bb_advisor = record.bb_advisor ;
    m_bb_cannon = record.bb_cannon;
    m_king_vertex = record.king_vertex;
    m_bb_color = record.bb_color;
}


bool Board::is_king_face_king() const {

    auto success = bool{false};
    const auto red_x = get_x(m_king_vertex[Types::RED]);
    const auto black_x = get_x(m_king_vertex[Types::BLACK]);
    if (red_x == black_x) {
        const auto file = static_cast<Types::File>(red_x);
        const auto file_bb = Utils::file2bitboard(file);
        const auto cnt = Utils::count_few(file_bb);
        if (cnt == 2) {
            success = true;
        }
    }
    return success;
}

bool Board::is_checkmate(const Types::Vertices vtx) const {
    const auto pt = get_piece_type(vtx);
    if (pt == Types::EMPTY_PIECE_T || 
            pt == Types::KING ||
            pt == Types::ELEPHANT ||
            pt == Types::ADVISOR) {
        return false;
    }

    const auto at = Utils::vertex2bitboard(vtx);
    const auto color = m_bb_color[Types::BLACK] & at ? Types::BLACK : Types::RED;
    const auto opp_king_bb = Utils::vertex2bitboard(m_king_vertex[swap_color(color)]);
    auto attack = BitBoard{};

    if (pt == Types::PAWN) {
        attack = m_pawn_attacks[color][vtx];
    } else if (pt == Types::HORSE) {
        const auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
        attack = m_horse_magics[vtx].attack(occupancy);
    } else if (pt == Types::CANNON) {
        const auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
        const auto rankattack = m_cannonrank_magics[vtx].attack(occupancy);
        const auto fileattack = m_cannonfile_magics[vtx].attack(occupancy);
        attack = rankattack | fileattack;
    } else if (pt == Types::ROOK) {
        const auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
        const auto rankattack = m_rookrank_magics[vtx].attack(occupancy);
        const auto fileattack = m_rookfile_magics[vtx].attack(occupancy);
        attack = rankattack | fileattack;
    }

    return opp_king_bb & attack;
}

bool Board::is_attack(const Types::Vertices vtx) const {
    auto pt = get_piece_type(vtx);
    if (pt == Types::EMPTY_PIECE_T) {
        return false;
    }

    const auto at = Utils::vertex2bitboard(vtx);
    const auto color = m_bb_color[Types::BLACK] & at ? Types::BLACK : Types::RED;
    auto attack = BitBoard{};

    if (pt == Types::KING) {
        attack = m_king_attacks[vtx];
    } else if (pt == Types::PAWN) {
        attack = m_pawn_attacks[color][vtx];
    } else if (pt == Types::ELEPHANT) {
        const auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
        attack = m_elephant_magics[vtx].attack(occupancy);
    } else if (pt == Types::ADVISOR) {
         attack = m_advisor_attacks[vtx];
    } else if (pt == Types::HORSE) {
        const auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
        attack = m_horse_magics[vtx].attack(occupancy);
    } else if (pt == Types::CANNON) {
        const auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
        const auto rankattack = m_cannonrank_magics[vtx].attack(occupancy);
        const auto fileattack = m_cannonfile_magics[vtx].attack(occupancy);
        attack = rankattack | fileattack;
    } else if (pt == Types::ROOK) {
        const auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
        const auto rankattack = m_rookrank_magics[vtx].attack(occupancy);
        const auto fileattack = m_rookfile_magics[vtx].attack(occupancy);
        attack = rankattack | fileattack;
    }

    return m_bb_color[swap_color(color)] & attack;
}

bool Board::is_legal(Move move) {
    auto movelist = std::vector<Move>{};
    generate_movelist(get_to_move(), movelist);
    auto success = bool{false};
    
    for (const auto &m : movelist) {
        if (move.get_data() == m.get_data()) {
            success = true;
            break;
        }
    }

    return success;
}

Move Board::text2move(std::string text) {
    if (text.size() != 4) {
        return Move{};
    }
    
    const auto str2vertex = [&](const char *s) -> Types::Vertices {
        char x_char = s[0];
        char y_char = s[1];
        int x = -1;
        int y = -1;
        
        if (x_char >= 'a' && x_char <= 'j') {
            x = static_cast<int>(x_char - 'a');
        }
        if (y_char >= '0' && y_char <= '9') {
            y = static_cast<int>(y_char - '0');
        }
        
        if (x == -1 || y == -1) {
            return Types::NO_VERTEX;
        }
        
        return static_cast<Types::Vertices>(get_vertex(x, y));
    };
    
    const auto from = str2vertex(text.data());
    const auto to = str2vertex(text.data() + 2);
    
    if (from == Types::NO_VERTEX || to == Types::NO_VERTEX) {
        return Move{};
    }

    return Move(from, to);
}

void Board::increment_movenum() {
    m_movenum++;
    m_gameply = (m_movenum / 2) + 1;
}

void Board::decrement_movenum() {
    m_movenum--;
    m_gameply = (m_movenum / 2) + 1;
}

bool Board::is_eaten() const {
    return m_eaten;
}

std::array<Types::Vertices, 2> Board::get_kings() const {
    return m_king_vertex;
}

Types::Color Board::get_to_move() const {
    return m_tomove;
}

int Board::get_movenum() const {
    return m_movenum;
}

int Board::get_gameply() const {
    return m_gameply;
}

std::uint64_t Board::get_hash() const {
    return m_hash;
}

Move Board::get_last_move() const {
    return m_lastmove;
}
