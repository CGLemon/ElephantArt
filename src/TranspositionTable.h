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

#ifndef TRANSPOSITIONTABLE_H_INCLUDE
#define TRANSPOSITIONTABLE_H_INCLUDE

#include <vector>

#include "BitBoard.h"

struct TTEntry {
    void save(TTEntry &&tte) {
        hash = tte.hash;
        move = tte.move;

        value = tte.value;
        eval = tte.eval;
        depth = tte.depth;
        generation = tte.generation;
    }

    void save(std::uint64_t h, Move m, int v,
                  int e, int d, int g) {
        hash = h;
        move = m;

        value = v;
        eval = e;
        depth = d;
        generation = g;
    }

    std::uint64_t hash{0ULL};

    Move move;
    int value;
    int eval;
    int depth;
    int generation{0};
};

class TranspositionTable {
public:
    TranspositionTable();

    bool probe(std::uint64_t hash, TTEntry &entry);
    void insert(std::uint64_t hash, TTEntry &entry);
    void clear();

    void set_memory(size_t MiB);
    void update_generation();

private:
    static constexpr size_t MAX_ENTRY_SIZE = 1000000;
    static constexpr size_t MIN_ENTRY_SIZE = 10000;
    static constexpr size_t ENTRY_MEM_SIZE = sizeof(TTEntry);

    static constexpr size_t CLUSTER_COUNT = 8;

    inline std::vector<TTEntry>::iterator get_first_tte(std::uint64_t hash) {
        auto idx = hash / m_cluster_size;
        return  std::begin(m_entry) + CLUSTER_COUNT * idx;
    }

    void resize(size_t size);

    std::vector<TTEntry> m_entry;
    size_t m_entry_size;
    size_t m_cluster_size;
};


#endif
