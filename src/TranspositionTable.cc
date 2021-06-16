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
    cluster_size = size/CLUSTER_COUNT;

    m_entry_size = size;
    m_cluster_size = cluster_size;
    m_entry.resize(m_entry_size);

    m_entry.shrink_to_fit();
}

TTEntry *TranspositionTable::probe(std::uint64_t hash, bool &hit) {
    TTEntry * first_tte = get_first_tte(hash);
    TTEntry * tte;

    int lowest_generation = std::numeric_limits<int>::max();
    int lowest_generation_index = 0;

    for (size_t i = 0; i < CLUSTER_COUNT; ++i) {
        tte = first_tte + i;

        if (tte->generation < lowest_generation) {
            lowest_generation = tte->generation;
            lowest_generation_index = i;
        }

        if (tte->hash == hash) {
            hit = true;
            return tte;
        }
    }

    hit = false;
    return first_tte + lowest_generation_index;
}
