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

#ifndef FORCEDCHECKMATE_H_INCLUDE
#define FORCEDCHECKMATE_H_INCLUDE

#include "Position.h"
#include "BitBoard.h"
#include "Types.h"

class ForcedCheckmate {
public:
    ForcedCheckmate(Position &position);

    Move find_checkmate();
    Move find_checkmate(std::vector<Move> &movelist);
    bool is_opp_checkmate();
    bool is_opp_checkmate(std::vector<Move> &movelist);

private:
    bool checkmate_search(Position &currentpos,
                          std::vector<std::uint64_t> &buf, int depth, int nodes) const;
    bool uncheckmate_search(Position &currentpos,
                            std::vector<std::uint64_t> &buf, int depth, int nodes) const;
    Position &m_rootpos;
    Types::Color m_color;
    int m_relaxed_move;
    int m_maxdepth;
    float m_factor;
};

#endif
