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

#include "Engine.h"
#include "config.h"

#include <sstream>

void Engine::initialize() {

    const auto games = (size_t)option<int>("num_games");

    while (m_positions.size() < games) {
        m_positions.emplace_back(std::make_shared<Position>());
    }

    while (m_positions.size() > games) {
        m_positions.pop_back();
    }

    int tag = 0;
    for (auto &p : m_positions) {
        p->init_game(tag);
        tag++;
    }

    if (m_network == nullptr) {
        m_network = std::make_shared<Network>();
        m_network->initialize(option<int>("playouts"), option<std::string>("weights_file"));
    }
}

int Engine::adj_position_ref(const int g) const {
    if (g < 0 || g > option<int>("num_games")) {
        return DEFUALT_POSITION;
    }
    return g;
}

std::shared_ptr<Position> Engine::get_position(const int g) const {
    const auto adj_g = adj_position_ref(g);
    return m_positions[adj_g];
}


void Engine::reset_game(const int g) {
    assert(g >= 0 || g < option<int>("num_games"));
    get_position(g)->init_game(g);
}


void Engine::display(const int g) const {
     get_position(g)->display();
}

std::vector<Move> Engine::get_movelist(const int g) const {
    return get_position(g)->get_movelist();
}

Engine::Response Engine::gather_movelist(const int g) const {

    auto rep = std::ostringstream{};
    const auto movelist = get_movelist(g);

    for (const auto &m: movelist) {
        rep << m.to_string() << " ";
    }

    return rep.str();
}

Engine::Response Engine::fen2board(std::string fen, const int g) {

    auto rep = std::ostringstream{};
    auto success = get_position(g)->fen2board(fen);
    if (success) {
        rep << "";
    } else {
        rep << "Illegal FEN format";
    }

    return rep.str();
}

Engine::Response Engine::do_textmove(std::string move, const int g) {

    auto rep = std::ostringstream{};
    auto success = get_position(g)->do_textmove(move);
    if (success) {
        rep << "";
    } else {
        rep << "Illegal move";
    }
    return rep.str();
}

Engine::Response Engine::undo_move(const int g) {
    auto rep = std::ostringstream{};
    auto success = get_position(g)->undo();
    if (success) {
        rep << "";
    } else {
        rep << "Fail to undo move";
    }
    return rep.str();
}

Engine::Response Engine::position(std::string fen,
                                  std::string moves, const int g) {

    auto rep = std::ostringstream{};
    auto success = get_position(g)->position(fen, moves);
    if (success) {
        rep << "";
    } else {
        rep << "Illegal position";
    }
    return rep.str();
}

