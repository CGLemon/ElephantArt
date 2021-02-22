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

#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

class TimeControl {
public:
    TimeControl(int milliseconds, int movestogo, int increment);
    int get_limittime() const;

    void set_lagbuffer(const int milliseconds);
    void set_plies(const int plies);
    void set_score(const int score);

private:
    int get_estimated_plies() const;

    int m_lagbuffer;
    int m_milliseconds;
    int m_movestogo;
    int m_increment;
    int m_plies;
    int m_score;
};

#endif
