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
        Utils::printf<Utils::SYNC>("Saya : ");

        auto input = std::string{};
        if (std::getline(std::cin, input)) {

            auto parser = Utils::CommandParser(input);
            Utils::printf<Utils::EXTERN>("%s\n", input.c_str());

            if (!parser.valid()) {
                Utils::printf<Utils::SYNC>(" No input command\n");
                continue;
            }

            if (parser.get_count() == 1 && parser.find("quit")) {
                Utils::printf<Utils::SYNC>("Exit\n");
                break;
            }
            auto out = execute(parser);
            Utils::printf<Utils::SYNC>("%s\n", out.c_str());
        }
    }
}

std::string ASCII::execute(Utils::CommandParser &parser) {
    auto out = std::ostringstream{};
    const auto lambda_syntax_not_understood =
        [&](Utils::CommandParser &p, size_t ignore) -> void {

        if (p.get_count() <= ignore) return;
        out << p.get_commands(ignore)->str << " ";
        out << ": syntax not understood" << std::endl;
    };

    if (const auto res = parser.find("legal-moves", 0)) {
        lambda_syntax_not_understood(parser, 1);
        const auto ascii_out = m_ascii_engine->gather_movelist();
        out << ascii_out << std::endl;
    } else if (const auto res = parser.find("fen", 0)) {
        lambda_syntax_not_understood(parser, 7);
        const auto cnt = parser.get_count();
        const auto limit = size_t{7};
        const auto max = std::min(cnt, limit);
        const auto fen = parser.get_slice(1, max)->str;
        const auto ascii_out = m_ascii_engine->fen(fen);
        out << ascii_out << std::endl;
    } else if (const auto res = parser.find("move", 0)) {
        lambda_syntax_not_understood(parser, 2);
        const auto move = parser.get_command(1)->str;
        const auto ascii_out = m_ascii_engine->do_textmove(move);
        out << ascii_out << std::endl;
    } else if (const auto res = parser.find("undo", 0)) {
        lambda_syntax_not_understood(parser, 1);
        const auto ascii_out = m_ascii_engine-> undo_move();
        out << ascii_out << std::endl;
    } else if (const auto res = parser.find("raw-nn", 0)) {
        lambda_syntax_not_understood(parser, 2);
        const auto cnt = parser.get_count();
        if (cnt == 1) {
            out << m_ascii_engine->raw_nn(0);
        } else {
            const auto symmetry = parser.get_command(1)->get<int>();
            out << m_ascii_engine->raw_nn(symmetry);
        }
    } else if (const auto res = parser.find("input-planes", 0)) {
        lambda_syntax_not_understood(parser, 2);
        const auto cnt = parser.get_count();
        if (cnt == 1) {
            out << m_ascii_engine->input_planes(0);
        } else {
            const auto symmetry = parser.get_command(1)->get<int>();
            out << m_ascii_engine->input_planes(symmetry);
        }
    } else if (const auto res = parser.find("history-board", 0)) {
        lambda_syntax_not_understood(parser, 1);
        out << m_ascii_engine->history_board();
    } else if (const auto res = parser.find("dump-maps", 0)) {
        lambda_syntax_not_understood(parser, 1);
        out << m_ascii_engine->get_maps();
    } else if (const auto res = parser.find({"genmove", "g"}, 0)) {
        lambda_syntax_not_understood(parser, 2);
        const auto cnt = parser.get_count();
        if (cnt == 1) { 
            out << m_ascii_engine->uct_move();
        } else {
            const auto mode = parser.get_command(1)->str;
            if (mode == "rand") {
                out << m_ascii_engine->rand_move();
            } else if (mode == "nn-direct") {
                out << m_ascii_engine->nn_direct_move();
            } else if (mode == "uct") {
                out << m_ascii_engine->uct_move();
            }
        }
    } else if (const auto res = parser.find("dump-collection", 0)) {
        lambda_syntax_not_understood(parser, 2);
        const auto cnt = parser.get_count();
        if (cnt == 1) { 
            out << m_ascii_engine->dump_collection();
        } else {
            const auto filename = parser.get_command(1)->str;
            out << m_ascii_engine->dump_collection(filename);
        }
    }  else if (const auto res = parser.find("self-play", 0)) {
        lambda_syntax_not_understood(parser, 1);
        out << m_ascii_engine->selfplay();
    } else if (const auto res = parser.find("position", 0)) {
        lambda_syntax_not_understood(parser, 1);
        auto pos = parser.get_commands(1)->str;
        out << m_ascii_engine->position(pos);
    } else if (const auto res = parser.find("printf-pgn", 0)) {
        lambda_syntax_not_understood(parser, 2);
        const auto cnt = parser.get_count();
        if (cnt == 1) {
            out << m_ascii_engine->printf_pgn();
        } else {
            const auto filename = parser.get_command(1)->str;
            out << m_ascii_engine->printf_pgn(filename);
        }
    } else {
        auto commands = parser.get_commands();
        out << "Unknown command: " << commands->str;
    }

    return out.str();
}

