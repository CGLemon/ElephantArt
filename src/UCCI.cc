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

UCCI::UCCI() {
    init();
    loop();
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
                Utils::printf<Utils::SYNC>("Exit\n");
                break;
            }
        }
    }
}
