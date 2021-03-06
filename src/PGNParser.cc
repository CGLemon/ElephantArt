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

void PGNParser::savepgn(std::string filename, Position &pos, PGNRecorder::Format_t fmt) const {
    auto pgn = from_position(pos, fmt);
    auto pgnstr = get_pgnstring(pgn);
    auto file = std::ofstream{};
    file.open(filename, std::ios::app);
    if (file.is_open()) {
        file << pgnstr;
        file.close();
    } else {
        Utils::printf<Utils::STATIC>("Couldn't Open file %s!\n", filename.c_str());
    }
}

void PGNParser::pgn_stream(std::ostream &out, Position &pos, PGNRecorder::Format_t fmt) const {
    auto pgn = from_position(pos, fmt);
    out << get_pgnstring(pgn);
}

void PGNParser::loadpgn(std::string filename, Position &pos) const {
    auto file = std::ifstream{};
    auto buffer = std::stringstream{};
    auto line = std::string{};

    file.open(filename);
    if (!file.is_open()) {
        Utils::printf<Utils::STATIC>("Could not opne file : %s!\n", filename.c_str());
        return;
    }

    while(std::getline(file, line)) {
        // We remove all the line break.
        buffer << line << " ";
    }
    file.close();

    auto pgns = std::vector<PGNRecorder>{};
    from_pgnfile(buffer, pgns);

    if (!pgns.empty()) {
        auto &pgn = pgns[0];
        pos.fen(pgn.start_fen);

        for (const auto &m: pgn.moves) {
            pos.do_move_assume_legal(m.second);
        }
    }
}

void PGNParser::gather_pgnlist(std::string filename, std::vector<PGNRecorder> &pgns) const {
    auto file = std::ifstream{};
    auto buffer = std::stringstream{};
    auto line = std::string{};

    file.open(filename);
    if (!file.is_open()) {
        Utils::printf<Utils::STATIC>("Could not opne file : %s!\n", filename.c_str());
        return;
    }

    while(std::getline(file, line)) {
        // We remove all the line break.
        buffer << line << " ";
    }
    file.close();
    from_pgnfile(buffer, pgns);
}

std::string PGNParser::get_pgnstring(PGNRecorder pgn) const {
    auto pgnstream = std::ostringstream{};

    const auto stream_helper = [](PGNRecorder &pgn, std::string flag, std::ostream &out) {
        auto ite = pgn.properties.find(flag);
        if (ite != std::end(pgn.properties)) {
            out << "[" << flag
                    << " \""
                    << ite->second
                    << "\"]"
                    << std::endl;
        }
    };

    stream_helper(pgn, "Game", pgnstream);
    stream_helper(pgn, "Event", pgnstream);
    stream_helper(pgn, "Site", pgnstream);
    stream_helper(pgn, "Date", pgnstream);
    stream_helper(pgn, "Round", pgnstream);
    stream_helper(pgn, "Red", pgnstream);
    stream_helper(pgn, "Black", pgnstream);
    stream_helper(pgn, "Result", pgnstream);
    stream_helper(pgn, "Opening", pgnstream);
    stream_helper(pgn, "FEN", pgnstream);
    stream_helper(pgn, "Format", pgnstream);

    auto fmt = pgn.format;
    auto cnt = 1;
    for (auto idx = size_t{0}; idx < pgn.moves.size();) {
        pgnstream << cnt++ << ". ";
        auto pair = pgn.moves[idx++];
        auto c1 = pair.first;
        if (c1 == Types::BLACK) {
            const auto spaces = fmt == PGNRecorder::ICCS ? 5 : 4;
            Utils::space_stream(pgnstream, spaces + 1);
        }

        auto m1 = pair.second;
        if (fmt == PGNRecorder::WXF) {
            pgnstream << Board::get_wxfstring(m1);
        } else if (fmt == PGNRecorder::ICCS) {
            pgnstream << Board::get_iccsstring(m1);
        }

        if (c1 == Types::BLACK) {
            pgnstream << std::endl;
            continue;
        }
        if (!(idx < pgn.moves.size())) {
            pgnstream << std::endl;
            break;
        }

        Utils::space_stream(pgnstream, 1);

        pair = pgn.moves[idx++];
        auto m2 = pair.second;
        if (fmt == PGNRecorder::WXF) {
            pgnstream << Board::get_wxfstring(m2);
        } else if (fmt == PGNRecorder::ICCS) {
            pgnstream << Board::get_iccsstring(m2);
        }
        pgnstream << std::endl;
    }

    pgnstream << pgn.properties["Result"] << std::endl;

    return pgnstream.str();
}

PGNRecorder PGNParser::from_position(Position &pos, PGNRecorder::Format_t fmt) const {
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

    auto pgn = PGNRecorder{};

    auto &history = pos.get_history();
    assert(!history.empty());

    pgn.start_fen = history[0]->get_fenstring();
    pgn.result = pos.get_winner(false);

    if (history.size() >= 2) {
        for (auto idx = size_t{1}; idx < history.size(); ++idx) {
            const auto to_move = history[idx-1]->get_to_move();
            const auto last_move = history[idx]->get_last_move();
            pgn.moves.emplace_back(to_move, last_move);
        }
    }

    pgn.properties.insert({"Game", "Chinese Chess"});
    pgn.properties.insert({"Event", PROGRAM + " vs. " + PROGRAM});
    pgn.properties.insert({"Site", "None"});
    pgn.properties.insert({"Date", "2021.01.01"});
    pgn.properties.insert({"Round", "None"});
    pgn.properties.insert({"Red", PROGRAM});
    pgn.properties.insert({"Black", PROGRAM});
    pgn.properties.insert({"Result", lambda_resultstring(pgn.result)});
    pgn.properties.insert({"Opening", "Unknown"});
    pgn.properties.insert({"ECCO", "Unknown"});
    pgn.properties.insert({"FEN", pgn.start_fen});
    if (fmt == PGNRecorder::WXF) {
        pgn.properties.insert({"Format", "WXF"});
    } else if (fmt == PGNRecorder::ICCS) {
        pgn.properties.insert({"Format", "ICCS"});
    }
    pgn.format = fmt;

    return pgn;
}

void PGNParser::from_pgnfile(std::istream &buffer, std::vector<PGNRecorder> &pgns) const {
    const auto lambda_property = [](std::istream &buf) -> auto {
        char c;
        auto name = std::ostringstream{};
        auto data = std::ostringstream{};
        auto quotation = false;
        while(buf.get(c)) {
            if (c == ']') {
                break;
            } else if (isspace(c) && !quotation) {
                continue;
            } else if (c == '"') {
                quotation = !(quotation);
                continue;
            }

            if (quotation) {
                data << c;
            } else {
                name << c;
            }
        }
        return std::make_pair<std::string, std::string>(name.str(), data.str());
    };

    const auto lambda_commit = [](std::istream &buf) -> auto {
        char c;
        auto commit = std::ostringstream{};
        while(buf.get(c)) {
            if (c == '}') {
                break;
            }
            commit << c;
        }
        return commit.str();
    };

    const auto lambda_move = [](std::string &move, size_t size, PGNRecorder::Format_t fmt) -> auto {
        auto m = Move{};
        if (fmt == PGNRecorder::WXF) {
            // Not complete.
        } else if (fmt == PGNRecorder::ICCS && size == 5) {
            int f_x = int(move[0]) - 65;
            int f_y = int(move[1]) - 48;
            int t_x = int(move[3]) - 65;
            int t_y = int(move[4]) - 48;
            auto fvtx = Board::get_vertex(f_x, f_y);
            auto tvtx = Board::get_vertex(t_x, t_y);
            m = Move(fvtx, tvtx);
        }
        return m;
    };

    auto error = false;
    auto not_support = false;
    auto cause = std::string{};

#define ERROR_HANDLE(proposition, why) \
    if (!(proposition)) {              \
        cause = why;                   \
        error = true;                  \
        break;                         \
    }

#define NOT_SUPPORT_HANDLE(proposition, why) \
    if (!(proposition)) {                    \
        cause = why;                         \
        not_support = true;                  \
        break;                               \
    }

    char c;
    auto cnt = size_t{0};
    auto property_step = false;
    auto pgn = std::make_shared<PGNRecorder>();
    auto temp_stream = std::ostringstream{};
    auto pos = std::make_shared<Position>();
    pgn = nullptr;

    while(buffer.get(c)) {
        if (c == '[') {
            if (!property_step) {
                if (pgn) {
                    pgns.emplace_back(*pgn);
                }
                pos->init_game(0);
                pgn = std::make_shared<PGNRecorder>();
                property_step = true;
            }
            pgn->properties.insert(lambda_property(buffer));
        } else if (c == '{') {
            lambda_commit(buffer); // unused
        } else if (isspace(c)) {
            if (cnt > 0) {
                auto temp = temp_stream.str();
                if (temp.find(".") != std::string::npos) {
                    // Do nothing.
                } else if (cnt == 4 || cnt == 5) {
                    auto move = lambda_move(temp, cnt, pgn->format);
                    auto to_move = pos->get_to_move();

                    ERROR_HANDLE(move.valid(), "Invalid Move List")
                    ERROR_HANDLE(pos->do_move(move), "Illegal Move List")

                    pgn->moves.emplace_back(to_move, move);
                } else if (cnt == 1 || cnt == 3 || cnt == 5) {
                    ERROR_HANDLE((pgn->properties["Result"] == temp), "Wrong Result");
                }
                temp_stream = std::ostringstream{};
                cnt = 0;
            }
        } else {
            if (property_step) {
                property_step = false;

                ERROR_HANDLE(pgn->properties.find("Format") != std::end(pgn->properties), "Lack of Format Lable");
                ERROR_HANDLE(pgn->properties.find("FEN") != std::end(pgn->properties), "Lack of FEN Lable");
                ERROR_HANDLE(pgn->properties.find("Result") != std::end(pgn->properties), "Lack of Result Lable");

                if (pgn->properties["Format"] == "WXF") {
                    pgn->format = PGNRecorder::WXF;
                    NOT_SUPPORT_HANDLE(false, "WXF Format Not Support");
                } else if (pgn->properties["Format"] == "ICCS") {
                    pgn->format = PGNRecorder::ICCS;
                } else {
                   ERROR_HANDLE(pos->fen(pgn->start_fen), "Illegal Format")
                }

                pgn->start_fen = pgn->properties["FEN"];
                ERROR_HANDLE(pos->fen(pgn->start_fen), "Illegal FEN Format")

                if (pgn->properties["Result"] == "1-0") {
                    pgn->result = Types::RED;
                } else if (pgn->properties["Result"] == "0-1") {
                   pgn->result = Types::BLACK;
                } else if (pgn->properties["Result"] == "1/2-1/2") {
                   pgn->result = Types::EMPTY_COLOR;
                } else if (pgn->properties["Result"] == "*") {
                   pgn->result = Types::INVALID_COLOR;
                } else {
                   ERROR_HANDLE(false, "Illegal Result Format")
                }
            }
            temp_stream << c;
            cnt++;
        }
    }

#undef ERROR_HANDLE
    if (pgn) {
        pgns.emplace_back(*pgn);
    }
    if (error) {
        Utils::printf<Utils::STATIC>("The PGN format is wrong! Games: %zu, Cause: %s.\n",
                                     pgns.size(), cause.c_str());
        pgns.clear();
    }
    if (not_support) {
        Utils::printf<Utils::STATIC>("The PGN format is not support! Games: %zu, Cause: %s.\n",
                                     pgns.size(), cause.c_str());
        pgns.clear();
    }
}
