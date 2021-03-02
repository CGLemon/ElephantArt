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

#ifndef ENGINE_H_INCLUDE
#define ENGINE_H_INCLUDE

#include "Position.h"
#include "Network.h"
#include "Search.h"
#include "Train.h"

#include <memory>
#include <string>

class Engine {
public:
    static constexpr int DEFUALT_POSITION = 0;

    using Response = std::string;
    
    void initialize();

    void reset_game(const int g = DEFUALT_POSITION);

    void display(const int g = DEFUALT_POSITION) const;

    Response gather_movelist(const int g = DEFUALT_POSITION);
    Response fen(std::string fen, const int g = DEFUALT_POSITION);
    Response do_textmove(std::string move, const int g = DEFUALT_POSITION);
    Response undo_move(const int g = DEFUALT_POSITION);
    Response position(std::string pos, const int g = DEFUALT_POSITION);
    Response raw_nn(const int g = DEFUALT_POSITION);
    Response input_planes(const int g = DEFUALT_POSITION);
    Response history_board(const int g = DEFUALT_POSITION);
    Response rand_move(const int g = DEFUALT_POSITION);
    Response nn_direct_move(const int g = DEFUALT_POSITION);
    Response uct_move(const int g = DEFUALT_POSITION);
    Response get_maps();
    Response dump_collection(std::string filename = "NO_FILE_NAME",
                             const int g = DEFUALT_POSITION);
    Response think(SearchSetting setting, const int g = DEFUALT_POSITION);
    Response interrupt(const int g = DEFUALT_POSITION);
    Response ponderhit(const int g = DEFUALT_POSITION);
    Response selfplay(const int g = DEFUALT_POSITION);
    Response printf_pgn(std::string filename = "NO_FILE_NAME", const int g = DEFUALT_POSITION);
private:
    int clamp(const int g) const;

    std::shared_ptr<Position> get_position(const int g) const;
    std::shared_ptr<Search> get_search(const int g) const;
    std::shared_ptr<Train> get_train(const int g) const;

    std::unique_ptr<Network> m_network{nullptr};
    std::vector<std::shared_ptr<Train>> m_train_group;
    std::vector<std::shared_ptr<Search>> m_search_group;
    std::vector<std::shared_ptr<Position>> m_positions;

};


#endif
