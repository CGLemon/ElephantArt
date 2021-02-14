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

#include "BitBoard.h"
#include "Utils.h"

void Utils::dump_bitboard(const BitBoard &bitboard, std::ostream &out) {
    const auto lambda_vertex = [&](int x_, int y_) -> int {
        return x_ + y_ * BITBOARD_SHIFT;
    };

    for (int y = 0; y < BITBOARD_HEIGHT; ++y) {
        for (int x = 0; x < BITBOARD_SHIFT; ++x) {
            const auto vertex = lambda_vertex(x, y);
            if (bitboard & (FirstPosition << vertex)) {
                out << "1";
            }
            else {
                out << "0";
            }

            if (x != BITBOARD_SHIFT) {
                out << " ";
            }
        }
        out << std::endl;
    }

    out << "Other :" <<std::endl;
    for (int v = BITBOARD_HEIGHT * BITBOARD_SHIFT; v < bitboard.width(); ++v) {
       if (bitboard & (FirstPosition << v)) {
           out << "1";
       }
       else {
           out << "0";
       }
    }
    out << std::endl;
}

void Utils::dump_bitboard(const BitBoard &bitboard) {
    auto out = std::ostringstream{};
    dump_bitboard(bitboard, out);
    Utils::printf<Utils::AUTO>(out);
}

Types::Vertices Move::get_from() const { 
    return static_cast<Types::Vertices>((m_data & FROM_MASK) >> 8); 
}

Types::Vertices Move::get_to() const { 
    return static_cast<Types::Vertices>(m_data & TO_MASK);
}

std::uint16_t Move::get_data() const {
    return m_data;
}

bool Move::valid() const {
    return m_data != INVALID;
}

bool Move::is_ok() const {
    return get_from() != get_to();
}

std::string Move::to_string() const {
    if (!valid()) {
        return std::string{"None"};
    }

    const auto lambda_parser = [](Types::Vertices vtx) -> std::string {
        auto lambda_out = std::ostringstream{};
        const auto v = static_cast<int>(vtx);
        const auto x = v % BITBOARD_SHIFT;
        const auto y = v / BITBOARD_SHIFT;

        lambda_out << static_cast<char>(x + 97);
        lambda_out << y;

        return lambda_out.str();
    };

    const auto from = get_from();
    const auto to = get_to();
 
    auto out = std::ostringstream{};
    out << lambda_parser(from);
    out << lambda_parser(to);
    return out.str();
}

bool Move::hit(BitBoard &b) const {

    if (!valid() || !is_ok()) {
        return false;
    }

    auto f_bitboard = get_from_bitboard();
    auto t_bitboard = get_to_bitboard();

    f_bitboard &= b;
    t_bitboard &= b;

    if (f_bitboard && t_bitboard) {
        return true;
    }
    return false;
}
