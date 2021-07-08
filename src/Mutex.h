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

#include <atomic>
#include <thread>
#include <mutex>
#include <cassert>

#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && \
    !defined(_M_ARM64)
#include <emmintrin.h>
#endif

// A very simple mutex.
class Mutex {
public:
    void lock() {
        // Test and Test-and-Set reduces memory contention
        // However, just trying to Test-and-Set first improves performance in almost
        // all cases.
        while (m_exclusive.exchange(true, std::memory_order_acquire)) {
            while (m_exclusive.load(std::memory_order_relaxed));
        }
    }

    void unlock() {
        auto lock_held = m_exclusive.exchange(false, std::memory_order_release);

        // If this fails it means we are unlocking an unlocked lock.
    #ifdef NDEBUG
        (void)lock_held;
    #else
        assert(lock_held);
    #endif
    }

private:
    std::atomic<bool> m_exclusive{false};
};

// A very simple spin lock.
class SpinMutex {
public:
    void lock(){
        int spins = 0;
        while (true) {
            int val = 0;
            if (m_exclusive.compare_exchange_weak(val, 1, std::memory_order_acq_rel)) {
                break;
            }

            // Help avoid complete resource starvation by yielding occasionally if
            // needed.
            if (++spins % 1024 == 0) {
               std::this_thread::yield();
            } else {
               spin_loop_pause();
            }
        }
    }

    void unlock() { m_exclusive.store(0, std::memory_order_release); }

private:
    inline void spin_loop_pause() {
#if !defined(__arm__) && !defined(__aarch64__) && !defined(_M_ARM) && \
    !defined(_M_ARM64)
        _mm_pause();
#endif
    }

    std::atomic<int> m_exclusive{0};
};
