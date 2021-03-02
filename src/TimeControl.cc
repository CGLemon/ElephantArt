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

#include "TimeControl.h"
#include "config.h"

TimeControl::TimeControl(int milliseconds, int movestogo, int increment) {
    m_lagbuffer = 0;
    m_plies = 0;
    m_milliseconds = milliseconds;
    m_movestogo = movestogo;
    m_increment = increment;
}

int TimeControl::get_limittime() const {
    if (m_movestogo <= 0 && m_increment <= 0) {
        auto remaining = static_cast<double>(m_milliseconds);
        auto estimated = static_cast<double>(get_estimated_plies());
        auto elasticity = 0.0f;
        return static_cast<int>(elasticity + remaining/estimated);
    }

    if (m_movestogo > 0) {
        auto remaining = static_cast<double>(m_milliseconds);
        auto movestogo = static_cast<double>(m_movestogo);
        auto elasticity = 0.0f;
        return static_cast<int>(elasticity + remaining/movestogo);
    }

    if (m_increment > 0) {
        auto remaining = static_cast<double>(m_milliseconds);
        auto increment = static_cast<double>(m_increment);
        auto estimated = static_cast<double>(get_estimated_plies());
        auto elasticity = 0.0f;
        return static_cast<int>(elasticity + (remaining + estimated * increment)/estimated);
    }

    return 0.f;
}

int TimeControl::get_estimated_plies() const {
    return m_maxplies - m_plies;
}

void TimeControl::set_score(const int score) {
    m_score = score;
}

void TimeControl::set_lagbuffer(const int milliseconds) {
    m_lagbuffer = milliseconds > 0 ? milliseconds : 0;
}

void TimeControl::set_plies(const int plies, const int draw_plies) {
    m_maxplies = draw_plies;
    m_plies = plies > 0 ? plies : 0;
}

