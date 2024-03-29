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

#include "UCCI.h"
#include "Search.h"

UCCI::UCCI() {
    init();
    loop();
}

void UCCI::init() {
    if (m_ucci_engine == nullptr) {
        m_ucci_engine = std::make_unique<Engine>();
    }
    m_ucci_engine->initialize();
}

void UCCI::loop() {
    while (true) {
        auto input = std::string{};
        if (std::getline(std::cin, input)) {

            auto parser = Utils::CommandParser(input);
            WRITING << ">>" << ' ' << input << std::endl;

            if (!parser.valid()) {
                continue;
            }

            if (parser.get_count() == 1 && parser.find("quit")) {
                m_ucci_engine->interrupt();
                break;
            }

            auto out = execute(parser);
            LOGGING << out;
        }
    }
}

std::string UCCI::execute(Utils::CommandParser &parser) {
    auto out = std::ostringstream{};
    if (const auto res = parser.find("ucci", 0)) {
        out << "id name " << PROGRAM << " " << VERSION << std::endl;
        out << "id copyright 2021 NA" << std::endl;
        out << "id author NA" << std::endl;
        out << "option usemillisec type check default "
                << Utils::bool_to_str(option<bool>("usemillisec")) << std::endl;
        out << "option cachesize type spin min 1 max 131072 default "
                << option<int>("cache_size") << std::endl;
        out << "option playouts type spin min 0 max 150000000 default "
                << option<int>("playouts") << std::endl;
        out << "ucciok" << std::endl;
    } else if (const auto res = parser.find("isready", 0)) {
        out << "readyok" << std::endl;
    } else if (const auto res = parser.find("uccinewgame", 0)) {
        m_ucci_engine->newgame();
    }else if (const auto res = parser.find({"display", "d"}, 0)) {
        m_ucci_engine->display();
    } else if (const auto res = parser.find("go", 0)) {
        auto setting = SearchSetting{};
        if (const auto ponder = parser.find("ponder")) {
            setting.ponder = true;
        }
        if (const auto draw = parser.find("draw")) {
            setting.draw = true;
        }
        if (const auto depth = parser.find_next("depth")) {
            setting.depth = depth->get<int>();
        }
        if (const auto nodes = parser.find_next("nodes")) {
            setting.nodes = nodes->get<int>();
        }
        if (const auto time = parser.find_next("time")) {
            if (option<bool>("usemillisec")) {
                setting.milliseconds = time->get<int>();
            } else {
                setting.milliseconds = 1000 * time->get<int>();
            }
        }
        if (const auto movestogo = parser.find_next("movestogo")) {
            setting.movestogo = movestogo->get<int>();
        }
        if (const auto increment = parser.find_next("increment")) {
            setting.increment = increment->get<int>();
        }
        if (const auto time = parser.find_next("opptime")) {
            // unused
        }
        if (const auto movestogo = parser.find_next("oppmovestogo")) {
            // unused
        }
        if (const auto increment = parser.find_next("oppincrement")) {
            // unused
        }
        m_ucci_engine->think(setting);
    } else if (const auto res = parser.find("stop", 0)) {
        m_ucci_engine->interrupt();
    } else if (const auto res = parser.find("ponderhit", 0)) {
        if (const auto draw = parser.find("draw")) {
            m_ucci_engine->ponderhit(true);
        } else {
            m_ucci_engine->ponderhit(false);
        }
    } else if (const auto res = parser.find("position", 0)) {
        auto pos = parser.get_commands(1)->str;
        m_ucci_engine->position(pos);
    } else if (const auto res = parser.find("setoption", 0)) {
        if (parser.get_count() >= 3) {
            const auto key = parser.get_command(1)->str;
            const auto val = parser.get_command(2)->str;
            m_ucci_engine->setoption(key, val);
        }
    } else {
        auto commands = parser.get_commands();
        out << "Unknown command: " << commands->str << std::endl;
    }
    return out.str();
}
