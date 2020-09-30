/*
    This file is part of Saya.
    Copyright (C) 2020 Hung-Zhe, Lin

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

#ifndef ENGINE_H_INCLUDE
#define ENGINE_H_INCLUDE

#include "Position.h"

#include <memory>

class Engine {
public:
    void init(int g = 1);

    void reset_game();

    void display() const;
    
private:
    std::vector<std::shared_ptr<Position>> m_positions;

    std::shared_ptr<Position> m_position{nullptr};

    size_t m_default{0};

};


#endif
