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

#include "Uint128_t.h"

#include <algorithm>
#include <utility>
#include <iomanip>
#include <sstream>

#define BIN_SCAN(NAME)                      \
for (auto i = size_t{0}; i < 64; ++i) {     \
    if ((NAME & (1ULL << (63 - i))) != 0) { \
      out << 1;                             \
    } else {                                \
      out << 0;                             \
    }                                       \
}

template<>
void Uint128_t::outStream<Uint128_t::Stream_t::BIN>(std::ostream &out) const {
    BIN_SCAN(UPPER);
    out << " | ";
    BIN_SCAN(LOWER);
}

#undef BIN_SCAN

template<>
void Uint128_t::outStream<Uint128_t::Stream_t::HEX>(std::ostream &out) const {
    out << std::setfill('0') << std::hex;
    out << std::setw(16) << UPPER;
    out << " | ";
    out << std::setw(16) << LOWER;
    out << std::setfill(' ') << std::dec;
}

void Uint128_t::dump_status() const {
    auto out = std::ostringstream{};
    out << "binary : ";
    outStream<Stream_t::BIN>(out);
    out << "\n";
  
    out << "hex    : ";
    outStream<Stream_t::HEX>(out);
    out << "\n";
    
    std::cout << out.str() << std::endl;
}

void Uint128_t::swap() {
    std::swap(UPPER, LOWER);
}
