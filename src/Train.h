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

#ifndef TRAIN_H_INCLUDE
#define TRAIN_H_INCLUDE

#include "config.h"
#include "Board.h"
#include "Model.h"
#include "UCTNode.h"
#include "Types.h"
#include "Position.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <list>

struct DataCollection {
    using MAPS_PAIR = std::pair<int, float>;
    using PIECES_PAIR = std::array<std::vector<int>, 2>;

    struct PositionPieces {
        PIECES_PAIR pawns;
        PIECES_PAIR horses;
        PIECES_PAIR cannons;
        PIECES_PAIR rooks;
        PIECES_PAIR elephants;
        PIECES_PAIR advisors;
        PIECES_PAIR kings;
    };

    using PIECES_HISTORY = std::vector<PositionPieces>;
    int version;
    int movenum;
    int gameply;
    int rule50_remaining;
    int repetitions;
    int moves_left{0};

    std::array<float, Board::NUM_INTERSECTIONS * INPUT_CHANNELS> input_planes;
    std::array<float, INPUT_FEATURES> input_features;

    PIECES_HISTORY pieces_history;

    std::vector<MAPS_PAIR> probabilities;
    Types::Piece_t piece;

    Types::Color to_move{Types::INVALID_COLOR};
    Types::Color winner{Types::INVALID_COLOR};

    void out_stream(std::ostream &out);
};

class Train {
public:
    Train();

    void gather_probabilities(UCTNode &node, Position &pos);
    void gather_move(Move move, Position &pos);

    void gather_winner(Types::Color color);
    void save_data(std::string filename, bool append = true);
    void data_stream(std::ostream &out);

    void clear_buffer();
    void supervised(std::string pgnfile, std::string datafile);

private:
    bool handle() const;
    int get_version() const;
    void push_buffer(DataCollection &data);

    using Step = std::shared_ptr<DataCollection>;
    std::list<Step> m_buffer;
    int m_counter;
    bool m_lock;
};

#endif
