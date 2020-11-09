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

#include "ASCII.h"

#include <functional>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>

ASCII::ASCII() {
    init();
    loop();
}

void ASCII::init() {
    if (m_ascii_engine == nullptr) {
        m_ascii_engine = std::make_unique<Engine>();
    }
    m_ascii_engine->initialize();
}

void ASCII::loop() {

    while (true) {

        m_ascii_engine->display();
        std::cout << "Saya : ";
        auto input = std::string{};

        if (std::getline(std::cin, input)) {

            auto parser = Utils::CommandParser(input);

            if (!parser.valid()) {
                std::cout << " No input command" << std::endl;
                continue;
            }

            if (parser.get_count() == 1 && parser.find("quit")) {
                std::cout << " exit " << std::endl;
                break;
            }

            std::cout << execute(parser) << std::endl;
        }
    }
}


std::string ASCII::execute(Utils::CommandParser &parser) {

    auto out = std::ostringstream{};
    const auto lambda_syntax_not_understood =
        [&](Utils::CommandParser &p, size_t ignore) -> void {

        if (p.get_count() <= ignore) { return; }
        out << p.get_commands(ignore)->str << " ";
        out << ": syntax not understood" << std::endl;
    };

    if (const auto res = parser.find("dump-legal-move", 0)) {

        lambda_syntax_not_understood(parser, 1);
        const auto ascii_out = m_ascii_engine->gather_movelist();
        out << ascii_out << std::endl;

    } else if (const auto res = parser.find("position", 0)) {
        lambda_syntax_not_understood(parser, 7);
        const auto cnt = parser.get_count();
        const auto limit = size_t{7};
        const auto max = std::min(cnt, limit);
        const auto fen = parser.get_slice(1, max)->str;
        const auto ascii_out = m_ascii_engine->fen2board(fen);
        out << ascii_out << std::endl;
        
    } else {
        out << "unknown command" << std::endl;
    }

    return out.str();
}

