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

#include "TranspositionTable.h"

#include <utility>
#include <limits>
#include <algorithm>

TranspositionTable::TranspositionTable() {
    resize(MIN_ENTRY_SIZE);
}

void TranspositionTable::clear() {
    std::for_each(std::begin(m_entry), std::end(m_entry),
        [](auto &tte) {
            tte.generation = 0;
            tte.hash = 0ULL;
        }
    );
}

void TranspositionTable::set_memory(size_t MiB) {
    resize(MiB * 1024 * 1024 / ENTRY_MEM_SIZE);
}

void TranspositionTable::resize(size_t size) {
    size_t cluster_size = (size / CLUSTER_COUNT) + bool(size % CLUSTER_COUNT);
    size = cluster_size * CLUSTER_COUNT;

    size = std::min(size, MAX_ENTRY_SIZE);
    size = std::max(size, MIN_ENTRY_SIZE);
    cluster_size = size/4;

    m_entry_size = size;
    m_cluster_size = cluster_size;
    m_entry.resize(m_entry_size);
    m_entry.shrink_to_fit();
}

bool TranspositionTable::probe(std::uint64_t hash, TTEntry &entry) {
    auto first_tte = get_first_tte(hash);

    for (size_t i = 0; i < CLUSTER_COUNT; ++i) {
        auto tte = first_tte + i;
        if (tte->hash == hash) {
            entry = *tte;
            return true;
        }
    }

    return false;
}

void TranspositionTable::insert(std::uint64_t hash, TTEntry &entry) {
    auto first_tte = get_first_tte(hash);
    auto lowest_generation = std::numeric_limits<int>::max();
    auto lowest_generation_index = 0;

    for (size_t i = 0; i < CLUSTER_COUNT; ++i) {
        auto tte = first_tte + i;
        // Find lowest generation entry.
        if (tte->generation < lowest_generation) {
            lowest_generation = tte->generation;
            lowest_generation_index = i;
        }
    }
    
    // Replace lowest generation entry.
    auto tte = first_tte + lowest_generation_index;
    tte->save(std::move(entry));
}
