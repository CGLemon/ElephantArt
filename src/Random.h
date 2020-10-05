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

#ifndef RANDON_H_INCLUDE
#define RANDON_H_INCLUDE

#include <cstdint>
#include <limits>
#include <random>
#include <thread>

/*
 * The basic methods follow th below github webside.
 * https://github.com/lemire/testingRNG
 */


/*
 * Select the different seed from different ways.
 * "THREADS_SEED" is default.
 */
static constexpr std::uint64_t THREADS_SEED = 0;
static constexpr std::uint64_t TIME_SEED = 1;

/*
 * Select the different random generator that you want.
 */
enum class random_t {
    SplitMix_64,
    XoroShiro128Plus,
};

template<random_t RandomType>
class Random {
public:
    Random() = delete;

    Random(std::uint64_t seed);

    static Random &get_Rng(const std::uint64_t seed = THREADS_SEED);

    // Get the random number
    std::uint64_t randuint64();

    template<int Range>
    std::uint32_t randfix() {
          static_assert(0 < Range && Range < std::numeric_limits<std::uint32_t>::max(),
                            "randfix out of range?\n");
          return static_cast<std::uint32_t>(randuint64()) % Range;
    }

    // It is the interface for STL.
    using result_type = std::uint64_t;

    constexpr static result_type min() {
        return std::numeric_limits<result_type>::min();
    }

    constexpr static result_type max() {
        return std::numeric_limits<result_type>::max();
    }

    result_type operator()() { return randuint64(); }

private:
    static constexpr size_t SEED_SZIE = 2;

    static thread_local std::uint64_t m_seeds[SEED_SZIE];

    void seed_init(std::uint64_t);
};

template<random_t T>
Random<T>::Random(std::uint64_t seed) {
    seed_init(seed);
}

template<random_t T>
Random<T> &Random<T>::get_Rng(const std::uint64_t seed) {
    static thread_local Random s_rng{seed};
    return s_rng;
}
#endif
