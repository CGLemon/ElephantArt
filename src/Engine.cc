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

    for (auto &p : m_positions) {
        p->init_game();
    }

    m_position = m_positions[m_default];
}

void Engine::reset_game() {
    if (m_position == nullptr) {
        return;
    }

    m_position->init_game();
}


void Engine::display() const {
    m_position->display();
}

std::vector<Move> Engine::get_movelist() const {
    return m_position->get_movelist();
}

Engine::Response Engine::gather_movelist() const {

    auto rep = std::ostringstream{};
    const auto movelist = m_position->get_movelist();

    for (const auto &m: movelist) {
        rep << m.to_string() << " ";
    }

    return rep.str();
}

Engine::Response Engine::fen2board(std::string fen) {

    auto rep = std::ostringstream{};
    auto success = m_position->fen2board(fen);
    if (success) {
        rep << "";
    } else {
        rep << "Illegal FEN format";
    }

    return rep.str();
}
