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

#include "Selfplay.h"
#include "config.h"
#include "Utils.h"
#include "Random.h"
#include "FileSystem.h"

#include <iomanip>
#include <sstream>

Selfplay::Selfplay() {
    init();
    loop();
}

void Selfplay::init() {
    if (m_selfplay_engine == nullptr) {
        m_selfplay_engine = std::make_unique<Engine>();
    }
    m_selfplay_engine->initialize();

    m_games.store(0);
    m_directory = option<std::string>("selfplay_directory");
    m_max_games = option<int>("selfplay_games");

    m_filename_hash = Random<random_t::XoroShiro128Plus>::get_Rng().randuint64();
    auto ss = std::ostringstream();

    ss << std::hex << m_filename_hash << std::dec;

    m_data_filename = option<std::string>("selfplay_directory") +
                          "/" + ss.str() + ".data";

    m_pgn_filename = option<std::string>("selfplay_directory") +
                          "/" + ss.str() + ".pgn";
}

bool Selfplay::handle() {
    if (m_directory.empty()) {
        ERROR << "No directory for saving data." << std::endl;
        ERROR << "Please add option --selfplay-directory <directory name>." << std::endl;
        return false;
    }

    if (m_max_games <= 0) {
        ERROR << "The number of self-play games is zero." << std::endl;
        ERROR << "Please add option --selfplay-games <integer>." << std::endl;
        return false;
    }
    return true;
}

void Selfplay::loop() {
    if (!handle()) {
        return;
    }

    // Dump some infomations.
    LOGGING << "Hash: " << m_filename_hash << std::endl;
    LOGGING << "Target self-play games: " << m_max_games << std::endl;
    LOGGING << "Directory for saving: " << m_directory  << std::endl;

    // If the directory didn't exist, creating a new one.
    if (!is_directory_exist(m_directory)) {
        create_directory(m_directory);
    }

    // Close all verbose.
    set_option("analysis-verbose", false);
    set_option("ucci_response", false);

    for (int g = 0; g < option<int>("sync_games"); ++g) {
        m_workers.emplace_back(
            [this, g]() -> void {
                auto main_thread = (g == 0);

                while (m_games.load() < m_max_games) {
                    m_selfplay_engine->selfplay(g);
                    {
                        std::lock_guard<std::mutex> lock(m_io_mtx);

                        // Save selfplay data.
                        m_selfplay_engine->dump_collection(m_data_filename, g);
                        m_selfplay_engine->printf_pgn(m_pgn_filename, g);
                    }
                    m_selfplay_engine->reset_game(g);
                    m_games.fetch_add(1);

                    if (main_thread) {
                        LOGGING << "Played " << m_games.load() << " games." << std::endl;
                    }
                }
            }
        );
    }

    for (auto &t : m_workers) {
        t.join();
    }
}
