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
    void out_stream(std::ostream &out) const;

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

inline Uint128_t Uint128_t::operator~() const {
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
    else if (shift > 0 && shift < 64) {
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
    else if (shift > 0 && shift < 64) {
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
#endif
