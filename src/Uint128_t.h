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

#ifndef UINT128_T_H_INCLUDE
#define UINT128_T_H_INCLUDE
#include <cstdint>
#include <iostream>

class Uint128_t {
private:
    std::uint64_t UPPER;
    std::uint64_t LOWER;

public:
    enum STREAM_T {
        BIN,
        HEX
    };

    constexpr Uint128_t()
               : UPPER(0ULL), LOWER(0ULL) {}

    constexpr Uint128_t(const Uint128_t &rhs)
                  : UPPER(rhs.UPPER), LOWER(rhs.LOWER) {}

    constexpr Uint128_t(Uint128_t &&rhs)
                  : UPPER(std::move(rhs.UPPER)), LOWER(std::move(rhs.LOWER)) {}

    constexpr Uint128_t(std::uint64_t upper, std::uint64_t lower)
                  : UPPER(upper), LOWER(lower) {}

    constexpr Uint128_t(std::uint64_t lower)
                  : UPPER(0ULL), LOWER(lower) {}



    int width() const {
        return 128;
    }

    inline std::uint64_t get_upper() const {
        return UPPER;
    }

    inline std::uint64_t get_lower() const {
        return LOWER;
    }


    template<STREAM_T T>
    void outStream(std::ostream &out) const;

    void dump_status() const;

    operator bool() const;

#define OPERATOR_TYPE(TYPE) \
operator TYPE() const

    OPERATOR_TYPE(std::uint8_t);
    OPERATOR_TYPE(std::uint16_t);
    OPERATOR_TYPE(std::uint32_t);
    OPERATOR_TYPE(std::uint64_t);
    
    OPERATOR_TYPE(char);
    OPERATOR_TYPE(short);
    OPERATOR_TYPE(int);
    OPERATOR_TYPE(long);
    OPERATOR_TYPE(long long);

#undef OPERATOR_TYPE
  
    Uint128_t &operator=(const Uint128_t &rhs);
    Uint128_t &operator=(Uint128_t &&rhs);
  
#define OPERATOR_BITWISE(BITWISE)                       \
Uint128_t operator BITWISE(const Uint128_t &rhs) const; \
Uint128_t &operator BITWISE##=(const Uint128_t & rhs);

    OPERATOR_BITWISE(&);
    OPERATOR_BITWISE(|);
    OPERATOR_BITWISE(^);
   
#undef OPERATOR_BITWISE

    Uint128_t operator~() const;

    Uint128_t operator+() const;
    Uint128_t operator-() const;
   
    Uint128_t operator<<(const int shift) const;
    Uint128_t &operator<<=(const int shift);
    Uint128_t operator>>(const int shift) const;
    Uint128_t &operator>>=(const int shift);
    
    bool operator!() const;
    bool operator&&(const Uint128_t &rhs) const;
    bool operator||(const Uint128_t &rhs) const;
    bool operator==(const Uint128_t &rhs) const;
    bool operator!=(const Uint128_t &rhs) const;
    bool operator>(const Uint128_t &rhs) const;
    bool operator<(const Uint128_t &rhs) const;
    bool operator>=(const Uint128_t &rhs) const;
    bool operator<=(const Uint128_t &rhs) const;

    Uint128_t operator+(const Uint128_t &rhs) const;
    Uint128_t operator+(const int i) const;
    Uint128_t &operator+=(const Uint128_t &rhs);
    
    Uint128_t operator-(const Uint128_t &rhs) const;
    Uint128_t operator-(const int i) const;
    Uint128_t &operator-=(const Uint128_t &rhs);
    
    Uint128_t operator*(const Uint128_t &rhs) const;
    Uint128_t &operator*=(const Uint128_t & rhs);
};

static Uint128_t tie(std::uint64_t upper, std::uint64_t lower) {
    return Uint128_t(upper, lower);
}

#endif
