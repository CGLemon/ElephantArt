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

#include "Utils.h"

Selfplay::Selfplay() {
    init();
    loop();
}

void Selfplay::init() {
    if (m_selfplay_engine == nullptr) {
        m_selfplay_engine = std::make_unique<Engine>();
    }
    m_selfplay_engine->initialize();
}

void Selfplay::loop() {
    m_games.store(0);

    LOGGING << "Target games: " << option<int>("selfplay_games") << std::endl;
    LOGGING << "Object directory: " << option<std::string>("selfplay_directory")  << std::endl;

    m_max_games =  option<int>("selfplay_games");
    m_data_filename = option<std::string>("selfplay_directory") + "/test.data";
    m_pgn_filename = option<std::string>("selfplay_directory") + "/test.pgn";

    set_option("ucci_response", false);

    for (int g = 0; g < option<int>("sync_games"); ++g) {
        m_workers.emplace_back(
            [this, g]() -> void {
                auto main_thread = (g == 0);

                while (m_games.load() < m_max_games) {
                    m_selfplay_engine->selfplay(g);
                    m_selfplay_engine->dump_collection(m_data_filename, g);

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
