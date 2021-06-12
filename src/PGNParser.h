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

#include "Position.h"
#include "BitBoard.h"
#include "Types.h"

#include <fstream>
#include <iostream>
#include <string>

struct PGNRecorder {
    enum Format_t { WXF, ICCS };

    using ColorMovePair = std::pair<Types::Color, Move>;
    Types::Color result{Types::INVALID_COLOR};

    std::unordered_map<std::string, std::string> properties;

    std::string start_fen;

    std::vector<ColorMovePair> moves;

    Format_t format;

    bool valid{false};
};

class PGNParser {
public:
    void savepgn(std::string filename, Position &pos, std::string fmt) const;
    void pgn_stream(std::ostream &out, Position &pos, std::string fmt) const;

    void loadpgn(std::string filename, Position &pos) const;
    void gather_pgnlist(std::string filename, std::vector<PGNRecorder> &pgns) const;

private:
    std::vector<std::string> chop_stream(std::istream &buffer) const;

    std::string get_pgnstring(PGNRecorder pgn) const;

    PGNRecorder from_position(Position &pos, PGNRecorder::Format_t fmt) const;

    PGNRecorder parse_pgnstring(std::string pgn, int idx) const;

    // void from_pgnfile(std::istream &buffer, std::vector<PGNRecorder> &pgns) const;
};
