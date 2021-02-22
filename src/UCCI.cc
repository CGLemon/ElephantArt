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
            Utils::printf<Utils::EXTERN>("%s\n", input.c_str());

            if (!parser.valid()) {
                continue;
            }

            if (parser.get_count() == 1 && parser.find("quit")) {
                Utils::printf<Utils::SYNC>("bye\n");
                break;
            }

            auto out = execute(parser);
            Utils::printf<Utils::SYNC>("%s", out.c_str());
        }
    }
}

std::string UCCI::execute(Utils::CommandParser &parser) {
    auto out = std::ostringstream{};
    if (const auto res = parser.find("ucci", 0)) {
        out << "id name " << PROGRAM << " " << VERSION << std::endl;
        out << "id author " << "NA" << std::endl;
        out << "ucciok" << std::endl;
    } else if (const auto res = parser.find("isready", 0)) {
        out << "readyok" << std::endl;
    } else if (const auto res = parser.find({"display", "d"}, 0)) {
        m_ucci_engine->display();
    } else if (const auto res = parser.find("go", 0)) {
        auto setting = SearchSetting{};
        if (const auto ponder = parser.find("ponder")) {
            setting.ponder = true;
        }
        if (const auto depth = parser.find_next("depth")) {
            setting.depth = depth->get<int>();
        }
        if (const auto nodes = parser.find_next("nodes")) {
            setting.nodes = nodes->get<int>();
        }
        if (const auto time = parser.find_next("time")) {
            setting.milliseconds = 1000 * time->get<int>();
        }
        if (const auto movestogo = parser.find_next("movestogo")) {
            setting.movestogo = movestogo->get<int>();
        }
        if (const auto increment = parser.find_next("increment")) {
            setting.increment = increment->get<int>();
        }
        out << m_ucci_engine->think(setting);
    } else if (const auto res = parser.find("stop", 0)) {
        m_ucci_engine->interrupt();
    } else if (const auto res = parser.find("ponderhit", 0)) {
        m_ucci_engine->ponderhit();
    } else if (const auto res = parser.find("position", 0)) {
        auto pos = parser.get_commands(1)->str;
        m_ucci_engine->position(pos);
    } else {
        auto commands = parser.get_commands();
        out << "Unknown command: " << commands->str << std::endl;
    }
    return out.str();
}
