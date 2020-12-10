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

#ifndef SHARED_MUTEX_H_INCLUDE
#define SHARED_MUTEX_H_INCLUDE

#include <atomic>
#include <thread>
#include <chrono>

class SharedMutex {
public:
    SharedMutex() {};

    void lock();
    void unlock();

    void lock_shared();
    void unlock_shared();

private:
    bool acquire_exclusive_lock();
    int get_share_counter();

    void take_break();

    std::atomic<int> m_share_counter{0};
    std::atomic<bool> m_exclusive{false};
    std::chrono::microseconds m_wait_microseconds{100};
};

inline void SharedMutex::take_break() {
    std::this_thread::sleep_for(m_wait_microseconds);
}

inline bool SharedMutex::acquire_exclusive_lock() {
    bool expected = false;
    return m_exclusive.compare_exchange_weak(expected, true);
}

inline int SharedMutex::get_share_counter() {
    return m_share_counter.load();
}

inline void SharedMutex::lock() {
    while (!acquire_exclusive_lock()) {
        take_break();
    }
    while (get_share_counter() > 0) {
        take_break();
    }
}

inline void SharedMutex::unlock() {
    auto v = m_exclusive.exchange(false);
    assert(v == true);
}

inline void SharedMutex::lock_shared() {
    while (!acquire_exclusive_lock()) {
        take_break();
    }
    m_share_counter.fetch_add(1);
    unlock();
}

inline void SharedMutex::unlock_shared() {
    m_share_counter.fetch_sub(1);
}

enum class lock_t {
    X_LOCK,
    S_LOCK
};

template<lock_t T>
class LockGuard {
public:
    LockGuard(SharedMutex &sm);
    ~LockGuard();

private:
    SharedMutex &m_sm;
};

template<> 
LockGuard<lock_t::X_LOCK>::LockGuard(SharedMutex &sm) : m_sm(sm) {
    m_sm.lock();
}

template<> 
LockGuard<lock_t::X_LOCK>:: ~LockGuard() {
    m_sm.unlock();
}

template<> 
LockGuard<lock_t::S_LOCK>::LockGuard(SharedMutex &sm) : m_sm(sm) {
    m_sm.lock_shared();
}

template<> 
LockGuard<lock_t::S_LOCK>:: ~LockGuard() {
    m_sm.unlock_shared();
}

#endif
