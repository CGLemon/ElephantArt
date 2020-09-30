/*
    This file is part of Saya.
    Copyright (C) 2020 Hung-Zhe, Lin

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

#include <string>
#include <sstream>
#include <iostream>

void ASCII::init() {
    if (m_ascii_engine == nullptr) {
        m_ascii_engine = std::make_unique<Engine>();
    }
    m_ascii_engine->init();
}

void ASCII::loop() {

    while (true) {
        m_ascii_engine->display();

        auto input = std::string{};
        std::cout << "Saya : ";

        if (std::getline(std::cin, input)) {

            auto parser = Utils::CommandParser(input);

            if (!parser.valid()) {
                std::cout << " No input command" << std::endl;
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

    if (parser.find("dump-legal-move")) {


    } else {
        out << "unknown command" << std::endl;
        // out << "syntax not understood" << std::endl;
    }

    return out.str();
}
