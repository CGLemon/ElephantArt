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

#ifndef ASCII_H_INCLUDE
#define ASCII_H_INCLUDE

#include "Engine.h"
#include "Utils.h"
#include "CMLInterface.h"

#include <memory>
#include <string>

class ASCII : public CMLInterface {
public:
    ASCII();
    ASCII(const ASCII&) = delete;
    ASCII& operator=(const ASCII&) = delete;

private:
    virtual void init();

    virtual void loop();

    virtual std::string execute(Utils::CommandParser &parser);

    std::unique_ptr<Engine> m_ascii_engine{nullptr};
};

#endif
