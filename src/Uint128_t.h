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
#include <utility>

class Uint128_t {
private:
    std::uint64_t UPPER;
    std::uint64_t LOWER;

public:
    enum class Stream_t {
        BIN,
        HEX
    };

    constexpr Uint128_t()
                  : UPPER(0ULL), LOWER(0ULL) {}

    constexpr Uint128_t(const Uint128_t &rhs)
                  : UPPER(rhs.UPPER), LOWER(rhs.LOWER) {}

    constexpr Uint128_t(const Uint128_t &&rhs)
                  : UPPER(std::move(rhs.UPPER)), LOWER(std::move(rhs.LOWER)) {}

    constexpr Uint128_t(std::uint64_t upper, std::uint64_t lower)
                  : UPPER(upper), LOWER(lower) {}

    constexpr Uint128_t(std::uint64_t lower)
                  : UPPER(0ULL), LOWER(lower) {}

    inline constexpr int width() const {
        return 128;
    }

    inline std::uint64_t get_upper() const {
        return UPPER;
    }

    inline std::uint64_t get_lower() const {
        return LOWER;
    }

    void swap();

    template<Stream_t T>
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

static constexpr Uint128_t tie(std::uint64_t upper, std::uint64_t lower) {
    return Uint128_t(upper, lower);
}

/*
 * ============================= Implement functions =================================
 * 
 */

static constexpr Uint128_t uint128_0(0ULL);
static constexpr Uint128_t uint128_1(1ULL);

inline Uint128_t::operator bool() const {
    return static_cast<bool>(UPPER | LOWER);
}

#define OPERATOR_TYPE(TYPE)                 \
inline Uint128_t::operator TYPE() const {   \
  return static_cast<TYPE>(LOWER);          \
}

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


inline Uint128_t &Uint128_t::operator=(const Uint128_t &rhs) {
    UPPER = rhs.UPPER;
    LOWER = rhs.LOWER;
    return *this;
}

inline Uint128_t &Uint128_t::operator=(Uint128_t &&rhs) {
    if (this != &rhs) {
        UPPER = std::move(rhs.UPPER);
        LOWER = std::move(rhs.LOWER);
        rhs.UPPER = 0ULL;
        rhs.LOWER = 0ULL;
    }
    return *this;
}

#define OPERATOR_BITWISE(BITWISE)                 \
inline Uint128_t Uint128_t::operator BITWISE(     \
              const Uint128_t &rhs) const {       \
    return Uint128_t(UPPER BITWISE rhs.UPPER,     \
                    LOWER BITWISE rhs.LOWER);     \
}                                                 \
inline Uint128_t &Uint128_t::operator BITWISE##=( \
               const Uint128_t &rhs){             \
    UPPER BITWISE##= rhs.UPPER;                   \
    LOWER BITWISE##= rhs.LOWER;                   \
    return *this;                                 \
}

    OPERATOR_BITWISE(&);
    OPERATOR_BITWISE(|);
    OPERATOR_BITWISE(^);

#undef OPERATOR_BITWISE

inline Uint128_t Uint128_t::operator ~() const {
    return Uint128_t(~UPPER, ~LOWER);
}

inline Uint128_t Uint128_t::operator+() const{
    return *this;
}

inline Uint128_t Uint128_t::operator-() const{
    return ~*this + uint128_1;
}


inline Uint128_t Uint128_t::operator<<(const int shift) const {

    if (shift == 0) {
        return *this;
    }
    else if (shift == 64) {
        return Uint128_t(LOWER, 0ULL);
    }
    else if (shift < 64) {
        const auto temp = LOWER >> (64 - shift);
        return Uint128_t((UPPER << shift) | temp, LOWER << shift);
    }
    else if (shift > 64 && shift < 128) {
        return Uint128_t(LOWER << (shift-64), 0ULL);
    }
    else {
        return uint128_0;
    }
}

inline Uint128_t &Uint128_t::operator<<=(const int shift) {
    *this = *this << shift;
    return *this;
}

inline Uint128_t Uint128_t::operator>>(const int shift) const {

    if (shift == 0) {
        return *this;
    }
    else if (shift < 64) {
        const auto temp = UPPER << (64 - shift);
        return Uint128_t(UPPER >> shift, temp | (LOWER >> shift));
    }
    else if (shift == 64) {
        return Uint128_t(0ULL, UPPER);
    }
    else if (shift > 64 && shift < 128) {
        return Uint128_t(0ULL, UPPER >> (shift-64));
    }
    else {
        return uint128_0;
    }
}

inline Uint128_t &Uint128_t::operator>>=(const int shift) {
    *this = *this >> shift;
    return *this;
}

inline bool Uint128_t::operator!() const {
    return !static_cast<bool>(UPPER | LOWER);
}

inline bool Uint128_t::operator&&(const Uint128_t &rhs) const {
    return (static_cast<bool>(*this) && rhs);
}

inline bool Uint128_t::operator||(const Uint128_t &rhs) const {
    return (static_cast<bool>(*this) || rhs);
}

inline bool Uint128_t::operator==(const Uint128_t &rhs) const {
    return ((UPPER == rhs.UPPER) && (LOWER == rhs.LOWER));
}

inline bool Uint128_t::operator!=(const Uint128_t &rhs) const {
    return ((UPPER != rhs.UPPER) | (LOWER != rhs.LOWER));
}

inline bool Uint128_t::operator>(const Uint128_t &rhs) const {
    return (UPPER > rhs.UPPER) ||
               ((LOWER > rhs.LOWER) && (UPPER == rhs.UPPER));
}

inline bool Uint128_t::operator<(const Uint128_t &rhs) const {
    return (UPPER < rhs.UPPER) ||
               ((LOWER < rhs.LOWER) && (UPPER == rhs.UPPER));
}

inline bool Uint128_t::operator>=(const Uint128_t &rhs) const {
    return (UPPER < rhs.UPPER) ||
               ((LOWER <= rhs.LOWER) && (UPPER == rhs.UPPER));
}

inline bool Uint128_t::operator<=(const Uint128_t &rhs) const {
    return (UPPER < rhs.UPPER) ||
               ((LOWER >= rhs.LOWER) && (UPPER == rhs.UPPER));
}

inline Uint128_t Uint128_t::operator+(const Uint128_t &rhs) const {
    std::uint64_t new_lower = LOWER + rhs.LOWER;
    std::uint64_t carry = static_cast<std::uint64_t>(new_lower < LOWER);
    return Uint128_t(UPPER + rhs.UPPER + carry, new_lower);
}

inline Uint128_t Uint128_t::operator+(const int i) const {
    std::uint64_t new_lower = LOWER + i;
    std::uint64_t carry = static_cast<std::uint64_t>(new_lower < LOWER);
    return Uint128_t(UPPER + carry, new_lower);
}

inline Uint128_t Uint128_t::operator-(const Uint128_t &rhs) const {
    std::uint64_t new_lower = LOWER - rhs.LOWER;
    std::uint64_t carry = static_cast<std::uint64_t>(new_lower > LOWER);
    return Uint128_t(UPPER - rhs.UPPER - carry, new_lower);
}

inline Uint128_t Uint128_t::operator-(const int i) const {
    std::uint64_t new_lower = LOWER - i;
    std::uint64_t carry = static_cast<std::uint64_t>(new_lower > LOWER);
    return Uint128_t(UPPER - carry, new_lower);
}

inline Uint128_t &Uint128_t::operator+=(const Uint128_t &rhs) {
    std::uint64_t new_lower = LOWER + rhs.LOWER;
    UPPER += rhs.UPPER + static_cast<std::uint64_t>(new_lower < LOWER);
    LOWER += new_lower;
    return *this;
}

inline Uint128_t &Uint128_t::operator-=(const Uint128_t &rhs) {
    std::uint64_t new_lower = LOWER - rhs.LOWER;
    UPPER -= rhs.UPPER - static_cast<std::uint64_t>(new_lower < LOWER);
    LOWER -= new_lower;
    return *this;
}

inline Uint128_t Uint128_t::operator*(const Uint128_t & rhs) const{
    // split values into 4 32-bit parts
    std::uint64_t top[4] = {UPPER >> 32, UPPER & 0xffffffff, LOWER >> 32, LOWER & 0xffffffff};
    std::uint64_t bottom[4] = {rhs.UPPER >> 32, rhs.UPPER & 0xffffffff, rhs.LOWER >> 32, rhs.LOWER & 0xffffffff};
    std::uint64_t products[4][4];

    // multiply each component of the values
    for(int y = 3; y > -1; --y){
        for(int x = 3; x > -1; --x){
            products[3 - x][y] = top[x] * bottom[y];
        }
    }

    // first row
    std::uint64_t fourth32 = (products[0][3] & 0xffffffff);
    std::uint64_t third32  = (products[0][2] & 0xffffffff) + (products[0][3] >> 32);
    std::uint64_t second32 = (products[0][1] & 0xffffffff) + (products[0][2] >> 32);
    std::uint64_t first32  = (products[0][0] & 0xffffffff) + (products[0][1] >> 32);

    // second row
    third32  += (products[1][3] & 0xffffffff);
    second32 += (products[1][2] & 0xffffffff) + (products[1][3] >> 32);
    first32  += (products[1][1] & 0xffffffff) + (products[1][2] >> 32);

    // third row
    second32 += (products[2][3] & 0xffffffff);
    first32  += (products[2][2] & 0xffffffff) + (products[2][3] >> 32);

    // fourth row
    first32  += (products[3][3] & 0xffffffff);

    // move carry to next digit
    third32  += fourth32 >> 32;
    second32 += third32  >> 32;
    first32  += second32 >> 32;

    // remove carry from current digit
    fourth32 &= 0xffffffff;
    third32  &= 0xffffffff;
    second32 &= 0xffffffff;
    first32  &= 0xffffffff;

    // combine components
    return Uint128_t((first32 << 32) | second32, (third32 << 32) | fourth32);
}

inline Uint128_t &Uint128_t::operator*=(const Uint128_t & rhs) {
    *this = *this * rhs;
    return *this;
}

#endif
