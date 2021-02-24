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

#ifndef UCCI_H_INCLUDE
#define UCCI_H_INCLUDE

/*
 * UCCI is a interface for chinese chess program. It is not for human.
 * If you want to know the detail of UCCI protocol, you can see below website.
 * https://www.xqbase.com/protocol/cchess_ucci.htm
 */

#include "Engine.h"
#include "Utils.h"
#include "CLInterface.h"

#include <memory>
#include <string>

class UCCI : public CLInterface {
public:
    UCCI();
    UCCI(const UCCI&) = delete;
    UCCI& operator=(const UCCI&) = delete;
     
private:
    virtual void init();

    virtual void loop();

    virtual std::string execute(Utils::CommandParser &parser);

    std::unique_ptr<Engine> m_ucci_engine{nullptr};

};

#endif
