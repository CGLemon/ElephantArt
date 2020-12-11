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

#ifndef CACHE_H_INCLUDE
#define CACHE_H_INCLUDE

#include <array>
#include <deque>
#include <memory>
#include <mutex>
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
    void resize(size_t size);

    void dump_stats();

    size_t get_estimated_size();

    void clear();

private:
    static constexpr size_t MAX_CACHE_COUNT = 150000;

    static constexpr size_t MIN_CACHE_COUNT = 6000;

    static constexpr size_t ENTRY_SIZE = sizeof(EntryType) +
                                         sizeof(std::uint64_t) +
                                         sizeof(std::unique_ptr<EntryType>);

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
bool Cache<EntryType>::lookup(std::uint64_t hash,
                                    EntryType &result) {

    // std::lock_guard<std::mutex> lock(m_mutex);
    LockGuard<lock_t::S_LOCK> lock(m_sm);
    
    bool success = true;
    ++m_lookups;

    const auto iter = m_cache.find(hash);
    if (iter == m_cache.end()) {
        success = false;
    } else {
        const auto &entry = iter->second;
        ++m_hits;
        result = entry->result;
    }
    return success;
}

template <typename EntryType>
void Cache<EntryType>::insert(std::uint64_t hash,
                                    const EntryType &result) {

    // std::lock_guard<std::mutex> lock(m_mutex);
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
void Cache<EntryType>::resize(size_t size) {

    m_size = (size > Cache::MAX_CACHE_COUNT ? Cache::MAX_CACHE_COUNT : 
              size < Cache::MIN_CACHE_COUNT ? Cache::MIN_CACHE_COUNT : size);

    // std::lock_guard<std::mutex> lock(m_mutex);
    LockGuard<lock_t::X_LOCK> lock(m_sm);
    
    while (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

template <typename EntryType> 
void Cache<EntryType>::clear() {

    // std::lock_guard<std::mutex> lock(m_mutex);
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
void Cache<EntryType>::dump_stats() {
    Utils::auto_printf("NNCache: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %lu size\n",
                       m_hits, m_lookups, 100. * m_hits / (m_lookups + 1), m_inserts,
                       m_cache.size());
}
#endif
