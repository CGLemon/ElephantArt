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

#ifndef SELFPLAY_H_INCLUDE
#define SELFPLAY_H_INCLUDE

#include "Engine.h"

#include <thread>
#include <memory>
#include <atomic>
#include <mutex>

class Selfplay {
public:
    Selfplay();
    Selfplay(const Selfplay&) = delete;
    Selfplay& operator=(const Selfplay&) = delete;

private:
    void init();

    bool handle();

    void loop();

    std::vector<std::thread> m_workers;
    std::unique_ptr<Engine> m_selfplay_engine{nullptr};

    std::atomic<int> m_games;
    int m_max_games;
    std::uint64_t m_filename_hash;

    std::string m_directory;
    std::string m_pgn_filename;
    std::string m_data_filename;
    std::mutex m_io_mtx;
};

#endif
