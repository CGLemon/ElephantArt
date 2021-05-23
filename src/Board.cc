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

#include "Board.h"
#include "Random.h"

#include <functional>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <iomanip>

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

#define PIECES_CACHE                                         \
const auto blk_pawn = option<char>("black_pawn_en");         \
const auto blk_cannon = option<char>("black_cannon_en");     \
const auto blk_rook = option<char>("black_rook_en");         \
const auto blk_horse = option<char>("black_horse_en");       \
const auto blk_elephant = option<char>("black_elephant_en"); \
const auto blk_advisor = option<char>("black_advisor_en");   \
const auto blk_king = option<char>("black_king_en");         \
const auto red_pawn = option<char>("red_pawn_en");           \
const auto red_cannon = option<char>("red_cannon_en");       \
const auto red_rook = option<char>("red_rook_en");           \
const auto red_horse = option<char>("red_horse_en");         \
const auto red_elephant = option<char>("red_elephant_en");   \
const auto red_advisor = option<char>("red_advisor_en");     \
const auto red_king = option<char>("red_king_en");

void Board::reset_board() {
    clear_status();
    auto start_position = get_start_position();
    fen2board(start_position);
    m_hash = calc_hash();
    m_bb_attacks = calc_attacks();
}

void Board::clear_status() {
    m_tomove = Types::RED;
    m_gameply = 0;
    m_movenum = 1;
    m_rule50_ply = 0;
    m_capture = false;
    m_lastmove = Move{};
    set_repetitions(0, 0);
}

void Board::init_symmetry() {
    const auto get_symmetry = [](const int x, const int y, bool symm) {
         int symm_x = x;
         int symm_y = y;
         if (symm) {
             symm_x = Board::WIDTH - x - 1;
         }
         return std::make_pair(symm_x, symm_y);
    };

    for (int symm = 0; symm < NUM_SYMMETRIES; ++symm) {
        for (int vtx = 0; vtx < NUM_VERTICES; ++vtx) {
            symmetry_nn_vtx_table[symm][vtx] = Types::NO_VERTEX;
        }
        for (int idx = 0; idx < INTERSECTIONS; ++idx) {
            symmetry_nn_idx_table[symm][idx] = 0;
        }
    }

    for (int symm = 0; symm < NUM_SYMMETRIES; ++symm) {
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                const auto sym_idx = get_symmetry(x, y, (bool)symm);
                const auto vtx = get_vertex(x, y);
                const auto idx = get_index(x, y);
                symmetry_nn_idx_table[symm][idx] =
                    get_index(sym_idx.first, sym_idx.second);
                symmetry_nn_vtx_table[symm][vtx] =
                    get_vertex(sym_idx.first, sym_idx.second);
            }
        }
    }
}

bool Board::fen2board(std::string &fen) {
    // FEN example:
    // rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1
    //
    // part 1: position
    // rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR
    //
    // part 2: to move
    // w
    //
    // part 3: invalid
    // -
    //
    // part 4: invalid
    // -
    //
    // part 5: invalid
    // 0
    //
    // part 6: move number
    // 1

    auto cnt_stream = std::stringstream{fen};
    const auto cnt = std::distance(std::istream_iterator<std::string>(cnt_stream),
                                   std::istream_iterator<std::string>());

    auto fen_format = std::stringstream{fen};
    auto fen_stream = std::string{};
    if (cnt == 1) {
        fen_format >> fen_stream;
        if (fen_stream == "startpos") {
            auto start_position = get_start_position();
            return fen2board(start_position);
        }
    }

    if (cnt != 6) {
        return false;
    }

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

    // part 1: position
    fen_format >> fen_stream;
    PIECES_CACHE;
    const auto bb_process = [](Types::Vertices &vtx, BitBoard &p_bb, BitBoard &c_bb) {
        auto bb = Utils::vertex2bitboard(vtx);
        p_bb |= bb;
        c_bb |= bb;
    };
    auto vtx = Types::VTX_A9;
    for (const char &c : fen_stream) {
        bool skip = false;
        if (c == blk_pawn) {
            bb_process(vtx, bb_pawn, bb_black);
        } else if (c == blk_cannon) {
            bb_process(vtx, bb_cannon, bb_black);
        } else if (c == blk_rook) {
            bb_process(vtx, bb_rook, bb_black);
        } else if (c == blk_horse) {
            bb_process(vtx, bb_horse, bb_black);
        } else if (c == blk_elephant) {
            bb_process(vtx, bb_elephant, bb_black);
        } else if (c == blk_advisor) {
            bb_process(vtx, bb_advisor, bb_black);
        } else if (c == blk_king) {
            king_vertex_black = vtx;
            bb_black |= Utils::vertex2bitboard(vtx);
        } else if (c == red_pawn) {
            auto bb = Utils::vertex2bitboard(vtx);
            bb_pawn |= bb;
            bb_red |= bb;
        } else if (c == red_cannon) {
            bb_process(vtx, bb_cannon, bb_red);
        } else if (c == red_rook) {
            bb_process(vtx, bb_rook, bb_red);
        } else if (c == red_horse) {
            bb_process(vtx, bb_horse, bb_red);
        } else if (c == red_elephant) {
            bb_process(vtx, bb_elephant, bb_red);
        } else if (c == red_advisor) {
            bb_process(vtx, bb_advisor, bb_red);
        } else if (c == red_king) {
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

    // part 2: to move
    fen_format >> fen_stream;
    auto tomove = Types::INVALID_COLOR;
    if (fen_stream == "w" || fen_stream == "r") {
        tomove = Types::RED;
    } else if (fen_stream == "b") {
        tomove = Types::BLACK;
    } else {
        success = false;
    }

    // part 3-5: invalid
    for (int k = 0; k < 3; ++k) {
        fen_format >> fen_stream;
    }

    // part 6: move number
    int movenum;
    fen_format >> movenum;
    if (movenum <= 0) {
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
        m_gameply = (movenum-1) * 2 + static_cast<int>(tomove);
        m_movenum = movenum;
        m_tomove = tomove;

        // Calculate the new hash value and attacks.
        m_hash = calc_hash();
        m_bb_attacks = calc_attacks();
    }

    return success;
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
    // Some masks may be out of the board (ex. VTX_J0, VTX_J1, ... ,VTX_J8, VTX_J9), or
    // out of itself legal area. Find them and set invalid.
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

        // The original hash table size is enough. But the adding the
        // addition size may be more easy to find all magic numbers.
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
            // Start to generate the magic number.
            if (b == begin) {
                // Initialize all datas.
                bit_iterator.set(0ULL);
                std::fill(std::begin(used), std::end(used), false);

                // TODO: Optimize the magic seed.
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
            // Reference(attack bitboard) is the targel which we want to be 
            // generated by magic number. 
            auto reference = generate_reference(center, occupancy);

            const auto index = magics[v].index(occupancy);
            if (!used[index]) {
                magics[v].attacks[index] = reference;
                used[index] = true;
            } else {
                // If the slot has been used. Mean the collision was happened. We
                // need to check if the slot is same as new reference. If they are
                // different. Fail this time.
                if (magics[v].attacks[index] != reference) {
                    b = begin-1;
                }
            }
        }
    };

    // Elephant reference generator.
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

    // Horse reference generator.
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

    // Rook reference generator.
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

    // Rook reference generator.
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

    // Cannon reference generator.
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

    // Cannon reference generator.
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

    auto timer = Utils::Timer{};

    // Initialize the magic tables.
    set_valid(m_elephant_magics);
    set_valid(m_horse_magics);
    set_valid(m_rookrank_magics);
    set_valid(m_rookfile_magics);
    set_valid(m_cannonrank_magics);
    set_valid(m_cannonfile_magics);

    for (auto v = Types::VTX_BEGIN; v < Types::VTX_END; ++v) {
        // generate magic tables.
        generate_magics(0, v, m_elephant_magics, elephant_reference);
        generate_magics(0, v, m_horse_magics, horse_reference);
        generate_magics(2, v, m_rookrank_magics, rookrank_reference);
        generate_magics(4, v, m_rookfile_magics, rookfile_reference);
        generate_magics(2, v, m_cannonrank_magics, cannonrank_reference);
        generate_magics(4, v, m_cannonfile_magics, cannonfile_reference);
    }

    const auto t = timer.get_duration();
    DEBUG << "Generating the Magic Numbers spent"
              << ' ' << t << ' ' << "second(s)" << std::endl;
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
    DEBUG << "Attacks Table Memory" << ' ' << ':' << ' '
              << ' ' << static_cast<float>(res)/(1024.f * 1024.f)
              << ' ' << "(Mib)" << std::endl;
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
    p == Types::R_CANNON    ? out << option<char>("red_cannon_en")   : p == Types::B_CANNON    ? out << option<char>("black_cannon_en")   :
    p == Types::R_ROOK      ? out << option<char>("red_rook_en")     : p == Types::B_ROOK      ? out << option<char>("black_rook_en")     :
    p == Types::R_HORSE     ? out << option<char>("red_horse_en")    : p == Types::B_HORSE     ? out << option<char>("black_horse_en")    :
    p == Types::R_ELEPHANT  ? out << option<char>("red_elephant_en") : p == Types::B_ELEPHANT  ? out << option<char>("black_elephant_en") :
    p == Types::R_ADVISOR   ? out << option<char>("red_advisor_en")  : p == Types::B_ADVISOR   ? out << option<char>("black_advisor_en")  :
    p == Types::R_KING      ? out << option<char>("red_king_en")     : p == Types::B_KING      ? out << option<char>("black_king_en")     :
    p == Types::EMPTY_PIECE ? out << " " : out << "error";
}

template<>
void Board::piece_stream<Types::CHINESE>(std::ostream &out, Types::Piece p) {
    using STR = std::string;
    p == Types::R_PAWN      ? out << option<STR>("red_pawn_ch")     : p == Types::B_PAWN      ? out << option<STR>("black_pawn_ch")     :
    p == Types::R_CANNON    ? out << option<STR>("red_cannon_ch")   : p == Types::B_CANNON    ? out << option<STR>("black_cannon_ch")   :
    p == Types::R_ROOK      ? out << option<STR>("red_rook_ch")     : p == Types::B_ROOK      ? out << option<STR>("black_rook_ch")     :
    p == Types::R_HORSE     ? out << option<STR>("red_horse_ch")    : p == Types::B_HORSE     ? out << option<STR>("black_horse_ch")    :
    p == Types::R_ELEPHANT  ? out << option<STR>("red_elephant_ch") : p == Types::B_ELEPHANT  ? out << option<STR>("black_elephant_ch") :
    p == Types::R_ADVISOR   ? out << option<STR>("red_advisor_ch")  : p == Types::B_ADVISOR   ? out << option<STR>("black_advisor_ch")  :
    p == Types::R_KING      ? out << option<STR>("red_king_ch")     : p == Types::B_KING      ? out << option<STR>("black_king_ch")     :
    p == Types::EMPTY_PIECE ? out << "  " : out << "error";
}

template<>
void Board::info_stream<Types::ASCII>(std::ostream &out) const {
    out << "{";
    if (m_tomove == Types::RED) {
        out << "Next player: RED";
    } else if (m_tomove == Types::BLACK) {
        out << "Next player: BLACK";
    }  else {
        out << "color error!";
    }

    out << ", Last move: " << get_last_move().to_string();
    out << ", Ply number: " << get_gameply();
    out << ", Fifty-Rule ply: " << get_rule50_ply();
    out << ", Hash: " << std::hex << get_hash() << std::dec;

    out << ",\n Fen: ";
    fen_stream(out);
    out << "}" << std::endl;
}

template<>
void Board::info_stream<Types::CHINESE>(std::ostream &out) const {
    out << "{";
    if (m_tomove == Types::RED) {
        out << "下一手：紅方";
    } else if (m_tomove == Types::BLACK) {
        out << "下一手：黑方";
    }  else {
        out << "color error!";
    }

    out << "，上一手棋：" << get_last_move().to_string();
    out << "，第 " << get_gameply() << " 手棋";
    out << ", 無吃子 " << get_rule50_ply() << " 手棋";
    out << "，哈希：" << std::hex << get_hash() << std::dec;

    out << "，\n Fen：";
    fen_stream(out);
    out << "}" << std::endl;
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

    out << " - - 0 " << get_movenum();
}

std::string Board::get_fenstring() const {
    auto out = std::ostringstream{};
    fen_stream(out);
    return out.str();
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
void Board::board_stream<Types::CHINESE>(std::ostream &out, const Move lastmove) const {
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
            piece_stream<Types::CHINESE>(out, coordinate_x, coordinate_y);
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

    info_stream<Types::CHINESE>(out);
}

std::uint64_t Board::calc_hash(const bool symm) const {
    auto res = Zobrist::zobrist_empty;

    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            const auto vtx = get_vertex(x, y);
            const auto pis = get_piece(vtx);
            const auto symm_vtx = symmetry_nn_vtx_table[int(symm)][vtx];
            if (is_on_board(vtx)) {
                res ^= Zobrist::zobrist[pis][symm_vtx];
            } 
        }
    }
    if (m_tomove == Types::RED) {
        res ^= Zobrist::zobrist_redtomove;
    }
    return res;
}

std::array<BitBoard, 2> Board::calc_attacks() {
// http://rportal.lib.ntnu.edu.tw/bitstream/20.500.12235/106625/1/n060147070s01.pdf
// According to "the design and implementation of the chinese chess program shark",
// computing the attack bit board helps the program to find out check move and threat
// move. It will be move efficiency to find out perpetual check and perpetual pursuit.

    auto bb_attacks = std::array<BitBoard, 2>{};
    auto movelist = std::vector<Move>{};
    bb_attacks[Types::RED] = generate_movelist(Types::RED, movelist);
    movelist.clear();
    bb_attacks[Types::BLACK] = generate_movelist(Types::BLACK, movelist);
    return bb_attacks;
}

bool Board::is_on_board(const Types::Vertices vtx) {
    return START_VERTICES[vtx] != Types::INVAL_PIECE;
}

std::string Board::get_start_position() {
    PIECES_CACHE
    
    auto start_pos = std::ostringstream{};
    start_pos << blk_rook << blk_horse << blk_elephant << blk_advisor
              << blk_king << blk_advisor << blk_elephant << blk_horse << blk_rook << "/";
    start_pos << "9/";
    start_pos << "1" << blk_cannon << "5" << blk_cannon << "1/";
    start_pos << blk_pawn << "1" << blk_pawn << "1" << blk_pawn << "1" << blk_pawn << "1" << blk_pawn << "/";
    start_pos << "9/";
    start_pos << "9/";
    start_pos << red_pawn << "1" << red_pawn << "1" << red_pawn << "1" << red_pawn << "1" << red_pawn << "/";
    start_pos << "1" << red_cannon << "5" << red_cannon << "1/";
    start_pos << "9/";
    start_pos << red_rook << red_horse << red_elephant << red_advisor
              << red_king << red_advisor << red_elephant << red_horse << red_rook;
    start_pos << " w - - 0 1";
    return start_pos.str();
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
                                          std::vector<Move> &movelist) -> void {
    while (legal_bitboard) {
        const auto res = Utils::extract(legal_bitboard);
        assert(res != Types::NO_VERTEX);

        const auto from = vtx;
        const auto to = res;
        movelist.emplace_back(Move(from, to));
    }
};

template<>
BitBoard Board::generate_move<Types::KING>(Types::Color color, std::vector<Move> &movelist) const {

    const auto vtx = m_king_vertex[color];
    const auto attack = m_king_attacks[vtx];
    const auto block = attack & m_bb_color[color];
    auto legal_bitboard = attack ^ block;

    lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
    return attack;
}

template<>
BitBoard Board::generate_move<Types::PAWN>(Types::Color color, std::vector<Move> &movelist) const {
    auto attacks = BitBoard(0ULL);
    auto bb_p = m_bb_pawn & m_bb_color[color];
    while (bb_p) {
        const auto vtx = Utils::extract(bb_p);
        const auto attack = m_pawn_attacks[color][vtx];
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
        attacks |= attack;
    }
    return attacks;
}

template<>
BitBoard Board::generate_move<Types::ADVISOR>(Types::Color color, std::vector<Move> &movelist) const {
    auto attacks = BitBoard(0ULL);
    auto bb_a = m_bb_advisor & m_bb_color[color];
    while (bb_a) {
        const auto vtx = Utils::extract(bb_a);
        const auto attack = m_advisor_attacks[vtx];
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
        attacks |= attack;
    }
    return attacks;
}

template<>
BitBoard Board::generate_move<Types::ELEPHANT>(Types::Color color, std::vector<Move> &movelist) const {
    auto attacks = BitBoard(0ULL);
    auto bb_e = m_bb_elephant & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    auto mask = color == Types::RED ? RedSide : BlackSide;
    while (bb_e) {
        const auto vtx = Utils::extract(bb_e);
        const auto attack = m_elephant_magics[vtx].attack(occupancy) & mask;
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = (attack ^ block);
        lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
        attacks |= attack;
    }
    return attacks;
}

template<>
BitBoard Board::generate_move<Types::HORSE>(Types::Color color, std::vector<Move> &movelist) const {
    auto attacks = BitBoard(0ULL);
    auto bb_h = m_bb_horse & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    while (bb_h) {
        const auto vtx = Utils::extract(bb_h);
        const auto attack = m_horse_magics[vtx].attack(occupancy);
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
        attacks |= attack;
    }
    return attacks;
}

template<>
BitBoard Board::generate_move<Types::ROOK>(Types::Color color, std::vector<Move> &movelist) const {
    auto attacks = BitBoard(0ULL);
    auto bb_r = m_bb_rook & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[swap_color(color)];
    while (bb_r) {
        const auto vtx = Utils::extract(bb_r);
        const auto rankattack = m_rookrank_magics[vtx].attack(occupancy);
        const auto fileattack = m_rookfile_magics[vtx].attack(occupancy);
        const auto attack = rankattack | fileattack;
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, movelist);
        attacks |= attack;
    }
    return attacks;
}

template<>
BitBoard Board::generate_move<Types::CANNON>(Types::Color color, std::vector<Move> &movelist) const {
    auto attacks = BitBoard(0ULL);
    auto opp_color = swap_color(color);
    auto bb_c = m_bb_cannon & m_bb_color[color];
    auto occupancy = m_bb_color[color] | m_bb_color[opp_color];
    while (bb_c) {
        const auto vtx = Utils::extract(bb_c);
        const auto rankattack = m_cannonrank_magics[vtx].attack(occupancy);
        const auto fileattack = m_cannonfile_magics[vtx].attack(occupancy);
        const auto attack = rankattack | fileattack;
        const auto block = attack & m_bb_color[color];
        auto legal_bitboard = attack ^ block;
        lambda_separate_bitboarad(vtx, legal_bitboard, movelist);

        // The cannons can not eat pieces directly.
        attacks |= (attack & m_bb_color[opp_color]);
    }
    return attacks;
}

BitBoard Board::generate_movelist(Types::Color color, std::vector<Move> &movelist) const {
    auto attacks = BitBoard(0ULL);

    // We don't remove the moves which may make the king be
    // killed. Not like chess game, to kill itself move is legal
    // at chinese chess.
    attacks |= generate_move<Types::PAWN>    (color, movelist);
    attacks |= generate_move<Types::CANNON>  (color, movelist);
    attacks |= generate_move<Types::ROOK>    (color, movelist);
    attacks |= generate_move<Types::HORSE>   (color, movelist);
    attacks |= generate_move<Types::ADVISOR> (color, movelist);
    attacks |= generate_move<Types::ELEPHANT>(color, movelist);
    attacks |= generate_move<Types::KING>    (color, movelist);

    if (is_king_face_king()) {
        const auto from = m_king_vertex[color];
        const auto to = m_king_vertex[swap_color(color)];
        const auto move = Move(from, to);
        movelist.emplace_back(move);
        attacks |= move.get_to_bitboard();
    }

    return attacks;
}

void Board::set_last_move(Move move) {
    m_lastmove = move;
}

void Board::set_to_move(Types::Color color) {
    update_zobrist_tomove(color, m_tomove);
    m_tomove = color;
}

void Board::set_repetitions(int repetitions, int cycle_length) {
    m_repetitions = repetitions;
    m_cycle_length = cycle_length;
}

void Board::swap_to_move() {
    set_to_move(swap_color(m_tomove));
}

void Board::do_move_assume_legal(Move move) {
    const auto from = move.get_from();
    const auto to = move.get_to();
    const auto form_bitboard = move.get_from_bitboard();
    const auto to_bitboard = move.get_to_bitboard();

    // Get the color.
    auto color = Types::INVALID_COLOR;
    if (m_bb_color[Types::BLACK] & form_bitboard) {
        color = Types::BLACK;
    } else if (m_bb_color[Types::RED] & form_bitboard) {
        color = Types::RED;
    }
    assert(color == get_to_move());

    // Get the piece type.
    const auto pt = get_piece_type(from);
    assert(pt != Types::EMPTY_PIECE_T);

    // Check whether the move capures the other.
    const auto opp_color = swap_color(color);
    m_capture = m_bb_color[opp_color] & to_bitboard;

    auto capture_pt = Types::EMPTY_PIECE_T;

    // Update it if the move capures the other.
    if (is_capture()) {
        capture_pt = get_piece_type(to);

        // Update bitboard.
        if (capture_pt == Types::KING) {
            m_king_vertex[opp_color] = Types::NO_VERTEX;
        } else {
            auto &ref_bb = get_piece_bitboard_ref(capture_pt);
            ref_bb ^= to_bitboard;
        }
        // Update color bitboard.
        m_bb_color[opp_color] ^= to_bitboard;
    }

    // Update bitboard.
    if (pt == Types::KING) {
        assert(m_king_vertex[color] == from);
        m_king_vertex[color] = to;
    } else {
        auto &ref_bb = get_piece_bitboard_ref(pt);
        ref_bb ^= form_bitboard;
        ref_bb ^= to_bitboard;
    }

    // Update color bitboard.
    m_bb_color[color] ^= form_bitboard;
    m_bb_color[color] ^= to_bitboard;

    auto p = static_cast<Types::Piece>(pt) + (color == Types::BLACK ? 7 : 0);

    // Update last move.
    set_last_move(move);

    // Update attacks
    m_bb_attacks = calc_attacks();

    // Update zobrist.
    update_zobrist(p , from, to);
    if (is_capture()) {
        auto capture_p = static_cast<Types::Piece>(capture_pt) + (color == Types::RED ? 7 : 0);
        update_zobrist_remove(capture_p, to);
    } 

    // Swap color.
    swap_to_move();

    // Increment move number.
    increment_gameply();

    // Increment rule50 ply.
    increment_rule50_ply();
    if (is_capture()) {
        set_rule50_ply(0);
    }
}

bool Board::is_king_face_king() const {
    const auto red_x = get_x(m_king_vertex[Types::RED]);
    const auto black_x = get_x(m_king_vertex[Types::BLACK]);
    if (red_x == black_x) {
        const auto mask = m_bb_color[Types::RED] | m_bb_color[Types::BLACK];
        const auto file = static_cast<Types::File>(red_x);
        const auto file_bb = Utils::file2bitboard(file);
        const auto cnt = Utils::count_few(file_bb & mask);
        if (cnt == 2) {
            return true;
        }
    }
    return false;
}

bool Board::is_check(const Types::Color color) const {
    const auto opp_color = swap_color(color);
    const auto opp_king = Utils::vertex2bitboard(m_king_vertex[opp_color]);
    const auto attacks = m_bb_attacks[color];

    return attacks & opp_king;
}

bool Board::is_legal(Move move) const {
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

void Board::increment_gameply() {
    m_movenum = ((++m_gameply)/2) + 1;
}

void Board::decrement_gameply() {
    m_movenum = ((--m_gameply)/2) + 1;
}

void Board::increment_rule50_ply() {
    ++m_rule50_ply;
}

void Board::set_rule50_ply(const int ply) {
    m_rule50_ply = ply;
}

bool Board::is_capture() const {
    return m_capture;
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

int Board::get_repetitions() const {
    return m_repetitions;
}

int Board::get_cycle_length() const {
    return m_cycle_length;
}

int Board::get_rule50_ply() const  {
    return m_rule50_ply;
}

int Board::get_rule50_ply_left() const  {
    return 100 - m_rule50_ply;
}

std::array<BitBoard, 2> Board::get_colors() const {
    return m_bb_color;
}

std::string Board::get_wxfstring(Move move) const {
    // We assune that it is not a variant.
    if (!is_legal(move)) {
        return std::string{"None"};
    }
    auto out = std::ostringstream{};

    auto redside = get_to_move() == Types::RED;
    const auto from_x = redside ? get_x(move.get_from()) : WIDTH  - get_x(move.get_from()) - 1;
    const auto from_y = redside ? get_y(move.get_from()) : HEIGHT - get_y(move.get_from()) - 1;
    const auto to_x = redside ? get_x(move.get_to()) : WIDTH  - get_x(move.get_to()) - 1;
    const auto to_y = redside ? get_y(move.get_to()) : HEIGHT - get_y(move.get_to()) - 1;

    const auto file = static_cast<Types::File>(from_x);
    const auto file_bb = Utils::file2bitboard(file);

    // const auto x_dis = to_x - from_x;
    const auto y_dis = to_y - from_y;

    const auto pt = get_piece_type(move.get_from());
    piece_stream<Types::ASCII>(out, static_cast<Types::Piece>(pt));

    if (pt == Types::KING || pt == Types::ADVISOR || pt == Types::ELEPHANT) {
        out << from_x + 1;
    } else if (pt == Types::HORSE || pt == Types::ROOK || pt == Types::CANNON) {
        const auto ref_bb = pt == Types::HORSE ? m_bb_horse :
                                pt == Types::ROOK ? m_bb_rook : m_bb_cannon;
        auto occupancy = ref_bb & m_bb_color[get_to_move()] & file_bb;
        const auto cnt = Utils::count_few(occupancy);
        assert(cnt == 1 || cnt == 2);

        if (cnt == 1) {
            out << from_x + 1;
        } else if (cnt == 2){
            auto low_vtx = Utils::extract(occupancy);
            auto high_vtx = Utils::extract(occupancy);
            if (!redside) {
                std::swap(low_vtx, high_vtx);
            }
            move.get_to() == high_vtx ? out << '+' : out << '.';
        }
    } else if (pt == Types::PAWN) {
        auto temp = std::vector<int>{};
        auto p_cnt = 0;
        auto p_occ = BitBoard(0ULL);
        for (auto ff = Types::FILE_A; ff < Types::FILE_J; ++ff) {
            const auto ff_bb = Utils::file2bitboard(file);
            auto occupancy = m_bb_pawn & m_bb_color[get_to_move()] & ff_bb;
            const auto cnt = Utils::count_few(occupancy);
            if (cnt >= 2) {
                temp.emplace_back(cnt);
            }
            if (ff == file) {
                p_cnt = cnt;
                p_occ = occupancy;
            }
        }

        if (p_cnt >= 2) {
            auto num = 0;
            while (p_occ) {
                num++;
                if (move.get_from() == Utils::extract(p_occ)) {
                    break;
                }
            }

            if (!redside) {
                num = p_cnt - num + 1;
            }

            auto tempout = std::ostringstream{};

            if (p_cnt == 2) {
                num == p_cnt ? tempout << '+' : tempout << '.';
            } else if (p_cnt == 3) {
                num == p_cnt ? tempout << '+' :
                    num == p_cnt-1 ? tempout << '-' : tempout << '.';
            } else {
                num == p_cnt ? tempout << '+' : tempout << static_cast<char>(97 + (p_cnt - num));
            }

            if (temp.size() >= 2) {
                out = std::ostringstream{};
                out << from_x + 1;
            }

            out << tempout.str();

        } else {
            out << from_x + 1;
        }
    } 

    if (pt == Types::PAWN || pt == Types::CANNON || pt == Types::ROOK || pt == Types::PAWN) {
        if (y_dis == 0) {
            out << '.' << to_x + 1;
        } else if (y_dis > 0) {
            out << '+'<< y_dis;
        } else if (y_dis < 0) {
            out << '-' << -y_dis;
        }
    } else if (pt == Types::ADVISOR || pt == Types::ELEPHANT || pt == Types::HORSE) {
        if (y_dis > 0) {
            out << '+' << to_x + 1;
        } else {
            out << '-' << to_x + 1;
        }
    }

    return out.str();
}

std::string Board::get_iccsstring(Move m) {
    return m.to_iccs();
}
