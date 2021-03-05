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

#include "PGNParser.h"
#include "Utils.h"
#include "config.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <cassert>

void PGNParser::save_pgn(std::string filename, Position &pos, PGNParser::Format_t fmt) {
    auto pgn = from_position(pos, fmt);
    auto file = std::ofstream{};
    file.open(filename, std::ios::out | std::ios::app);
    if (file.is_open()) {
        file << pgn;
        file.close();
    } else {
        Utils::printf<Utils::STATIC>("Couldn't Open file %s\n", filename.c_str());
    }
}

void PGNParser::pgn_stream(std::ostream &out, Position &pos, PGNParser::Format_t fmt) {
    out << from_position(pos, fmt);
}

std::string PGNParser::from_position(Position &pos, PGNParser::Format_t fmt) const {
    auto pgn = std::ostringstream{};
    auto pgnfmt = position_to_pgnformat(pos, fmt);

    const auto stream_helper = [](PGNFormat &pgnfmt, std::string flag, std::ostream &out) {
        auto ite = pgnfmt.properties.find(flag);
        if (ite != std::end(pgnfmt.properties)) {
            out << "[" << flag
                    << " \""
                    << ite->second
                    << "\"]"
                    << std::endl;
        }
    };

    stream_helper(pgnfmt, "Game", pgn);
    stream_helper(pgnfmt, "Event", pgn);
    stream_helper(pgnfmt, "Site", pgn);
    stream_helper(pgnfmt, "Date", pgn);
    stream_helper(pgnfmt, "Round", pgn);
    stream_helper(pgnfmt, "Red", pgn);
    stream_helper(pgnfmt, "Black", pgn);
    stream_helper(pgnfmt, "Result", pgn);
    stream_helper(pgnfmt, "Opening", pgn);
    stream_helper(pgnfmt, "FEN", pgn);
    stream_helper(pgnfmt, "Format", pgn);

    auto cnt = 1;
    for (auto idx = size_t{0}; idx < pgnfmt.moves.size();) {
        pgn << cnt++ << ". ";
        auto pair = pgnfmt.moves[idx++];
        auto c1 = pair.first;
        if (c1 == Types::BLACK) {
            const auto spaces = fmt == ICCS ? 5 : 4;
            Utils::space_stream(pgn, spaces + 2);
        }

        auto m1 = pair.second;
        if (fmt == WXF) {
            pgn << Board::get_wxfstring(m1);
        } else if (fmt == ICCS) {
            pgn << Board::get_iccsstring(m1);
        }

        if (c1 == Types::BLACK) {
            pgn << std::endl;
            continue;
        }
        if (!(idx < pgnfmt.moves.size())) {
            pgn << std::endl;
            break;
        }

        Utils::space_stream(pgn, 2);

        pair = pgnfmt.moves[idx++];
        auto m2 = pair.second;
        if (fmt == WXF) {
            pgn << Board::get_wxfstring(m2);
        } else if (fmt == ICCS) {
            pgn << Board::get_iccsstring(m2);
        }
        pgn << std::endl;
    }

    pgn << pgnfmt.properties["Result"] << std::endl;

    return pgn.str();
}

PGNFormat PGNParser::position_to_pgnformat(Position &pos, PGNParser::Format_t fmt) const {
    const auto lambda_resultstring = [](Types::Color color) -> std::string {
        if (color == Types::RED) {
            return std::string{"1-0"};
        } else if (color == Types::BLACK) {
            return std::string{"0-1"};
        } else if (color == Types::EMPTY_COLOR) {
            return std::string{"1/2-1/2"};
        }
        return std::string{"*"};
    };

    auto pgnfmt = PGNFormat{};

    auto &history = pos.get_history();
    assert(!history.empty());

    pgnfmt.start_fen = history[0]->get_fenstring();
    pgnfmt.result = pos.get_winner(false);

    if (history.size() >= 2) {
        for (auto idx = size_t{1}; idx < history.size(); ++idx) {
            const auto to_move = history[idx-1]->get_to_move();
            const auto last_move = history[idx]->get_last_move();
            pgnfmt.moves.emplace_back(to_move, last_move);
        }
    }

    pgnfmt.properties.insert({"Game", "Chinese Chess"});
    pgnfmt.properties.insert({"Event", PROGRAM + " vs. " + PROGRAM});
    pgnfmt.properties.insert({"Site", "None"});
    pgnfmt.properties.insert({"Date", "2021.01.01"});
    pgnfmt.properties.insert({"Round", "None"});
    pgnfmt.properties.insert({"Red", PROGRAM});
    pgnfmt.properties.insert({"Black", PROGRAM});
    pgnfmt.properties.insert({"Result", lambda_resultstring(pgnfmt.result)});
    pgnfmt.properties.insert({"Opening", "Unknown"});
    pgnfmt.properties.insert({"ECCO", "Unknown"});
    pgnfmt.properties.insert({"FEN", pgnfmt.start_fen});
    if (fmt == WXF) {
        pgnfmt.properties.insert({"Format", "WXF"});
    } else if (fmt == ICCS) {
        pgnfmt.properties.insert({"Format", "ICCS"});
    } 
    return pgnfmt;
}

