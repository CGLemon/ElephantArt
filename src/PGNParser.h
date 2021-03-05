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

struct PGNFormat {
    using ColorMovePair = std::pair<Types::Color, Move>;
    Types::Color result{Types::INVALID_COLOR};

    std::unordered_map<std::string, std::string> properties;

    std::string start_fen;

    std::vector<ColorMovePair> moves;
};

class PGNParser {
public:
    enum Format_t { WXF, ICCS };

    void save_pgn(std::string filename, Position &pos, Format_t fmt = ICCS);
    void pgn_stream(std::ostream &out, Position &pos, Format_t fmt = ICCS);

private:
    std::string from_position(Position &pos, Format_t fmt) const;
    PGNFormat position_to_pgnformat(Position &pos, Format_t fmt) const;
};
