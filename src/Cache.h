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

#ifndef CACHE_H_INCLUDE
#define CACHE_H_INCLUDE

#include <array>
#include <deque>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <cassert>

#include "Utils.h"
#include "SharedMutex.h"

template <typename EntryType>
class Cache {
public:
    Cache() : m_hits(0), m_lookups(0), m_inserts(0) {}

    bool lookup(std::uint64_t hash, EntryType &result);
    void insert(std::uint64_t hash, const EntryType &result);
    void set_memory(size_t MiB);
    void set_playouts(size_t playouts);

    size_t get_estimated_size();

    void clear();
    void clear_stats();

private:
    static constexpr size_t MAX_CACHE_MEM = 128 * 1024; // ~128 GiB

    static constexpr size_t MIN_CACHE_MEM = 1;

    static constexpr size_t ENTRY_SIZE = sizeof(EntryType) +
                                             sizeof(std::uint64_t) +
                                             sizeof(std::unique_ptr<EntryType>);

    void resize(size_t size);

    SharedMutex m_sm;
    size_t m_size;

    int m_hits;
    int m_lookups;
    int m_inserts;

    struct Entry {
        Entry(const EntryType &r) : result(r) {}
        EntryType result;
    };

    std::unordered_map<std::uint64_t, std::unique_ptr<const Entry>> m_cache;
    std::deque<std::uint64_t> m_order;
};

template <typename EntryType>
bool Cache<EntryType>::lookup(std::uint64_t hash, EntryType &result) {
    LockGuard<lock_t::S_LOCK> lock(m_sm);

    ++m_lookups;

    const auto iter = m_cache.find(hash);
    if (iter != m_cache.end()) {
        const auto &entry = iter->second;
        ++m_hits;
        result = entry->result;
        return true;
    }
    return false;
}

template <typename EntryType>
void Cache<EntryType>::insert(std::uint64_t hash, const EntryType &result) {
    LockGuard<lock_t::X_LOCK> lock(m_sm);
    
    if (m_cache.find(hash) == m_cache.end()) {
        m_cache.emplace(hash, std::make_unique<Entry>(result));
        m_order.emplace_back(hash);
        ++m_inserts;

        if (m_order.size() > m_size) {
            m_cache.erase(m_order.front());
            m_order.pop_front();
        }
    }
}

template <typename EntryType>
void Cache<EntryType>::set_memory(size_t MiB) {
    MiB = MiB > MAX_CACHE_MEM ? MAX_CACHE_MEM :
              MiB < MIN_CACHE_MEM ? MIN_CACHE_MEM : MiB;
    const double bytes = 1024.f * 1024.f * (double)MiB;
    const auto size = size_t(bytes / Cache::ENTRY_SIZE);
    resize(size);
}

template <typename EntryType>
void Cache<EntryType>::set_playouts(size_t playouts) {
    auto mem_used = (double)(playouts * Cache::ENTRY_SIZE) / (1024.f * 1024.f);
    set_memory((size_t)mem_used);
}

template <typename EntryType>
void Cache<EntryType>::resize(size_t size) {
    LockGuard<lock_t::X_LOCK> lock(m_sm);

    m_size = size;

    while (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

template <typename EntryType> 
void Cache<EntryType>::clear() {
    LockGuard<lock_t::X_LOCK> lock(m_sm);
    
    if (!m_order.empty()) {
        m_cache.clear();
        m_order.clear();
    }
}

template <typename EntryType>
size_t Cache<EntryType>::get_estimated_size() {
    return m_order.size() * Cache::ENTRY_SIZE;
}

template <typename EntryType>
void Cache<EntryType>::clear_stats() {
    LockGuard<lock_t::X_LOCK> lock(m_sm);
    m_hits = 0;
    m_lookups = 0;
    m_inserts = 0;
}
#endif
