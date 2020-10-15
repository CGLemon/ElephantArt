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
#include "Random.h"
#include <chrono>

namespace random_utils {

static inline std::uint64_t splitmix64(std::uint64_t z) {
    /*
     * The detail of parameteres are from
     * https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
     */

    z += 0x9e3779b97f4a7c15;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

static inline std::uint64_t rotl(const std::uint64_t x, const int k) {
    return (x << k) | (x >> (64 - k));
}

static inline std::uint64_t get_seed(std::uint64_t seed) {
    if (seed == THREADS_SEED) {
        // Get the seed from thread id.
        auto thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
        seed = static_cast<std::uint64_t>(thread_id);
    } else if (seed == TIME_SEED) {
        // Get the seed from system time.
        auto get_time = std::chrono::system_clock::now().time_since_epoch().count();
        seed = static_cast<std::uint64_t>(get_time);
    }
    return seed;
}
} // namespace random_utils


#define __RANDOM_INIT(TYPE__, CNT__)                 \
template<>                                           \
void Random<TYPE__>::seed_init(std::uint64_t seed) { \
    seed = random_utils::get_seed(seed);             \
    static_assert(SEED_SZIE >= CNT__,                \
        "The number of seeds is not enough?\n");     \
    for (auto i = size_t{0}; i < SEED_SZIE; ++i) {   \
        seed = random_utils::splitmix64(seed);       \
        Random<TYPE__>::m_seeds[i] = seed;           \
    }                                                \
}


template<random_t T>
thread_local std::uint64_t 
    Random<T>::m_seeds[Random<T>::SEED_SZIE];

__RANDOM_INIT(random_t::SplitMix_64, 1);

__RANDOM_INIT(random_t::XoroShiro128Plus, 2);

template<>
std::uint64_t Random<random_t::SplitMix_64>::randuint64() {
    /*
     * The detail of parameteres are from
     * https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
     */
    static constexpr size_t seed_Idx = SEED_SZIE - 1;

    Random<random_t::SplitMix_64>::m_seeds[seed_Idx] += 0x9e3779b97f4a7c15;
    auto z = Random<random_t::SplitMix_64>::m_seeds[seed_Idx];
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

template<>
std::uint64_t Random<random_t::XoroShiro128Plus>::randuint64() {
    /*
     * The detail of parameteres are from
     * https://github.com/lemire/testingRNG/blob/master/source/xoroshiro128plus.h
     */

    const std::uint64_t s0 = Random<random_t::XoroShiro128Plus>::m_seeds[0];
    std::uint64_t s1 = Random<random_t::XoroShiro128Plus>::m_seeds[1];
    const std::uint64_t result = s0 + s1;

    s1 ^= s0;
    Random<random_t::XoroShiro128Plus>::m_seeds[0] = 
        random_utils::rotl(s0, 55) ^ s1 ^ (s1 << 14);
    Random<random_t::XoroShiro128Plus>::m_seeds[1] = 
        random_utils::rotl(s1, 36);

    return result;
}
