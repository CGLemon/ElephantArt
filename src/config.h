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

#ifndef CONFIG_H_INCLUDE
#define CONFIG_H_INCLUDE

#include <string>
#include <unordered_map>

#define MARCRO_WIDTH 9
#define MARCRO_HEIGHT 10
#define MARCRO_SHIFT 10

const std::string PROGRAM = "ElephantArt";

const std::string VERSION = "pre-alpha"; 

const std::string NO_WEIGHT_FILE_NAME = "NO_WEIGHT_FILE";

const std::string NO_LOG_FILE_NAME = "NO_LOG_FILE";

template<typename T>
T option(std::string name);

template<typename T>
bool set_option(std::string name, T val);

void init_basic_parameters();

class ArgsParser {
public:
    ArgsParser() = delete;
    ArgsParser(int argc, char** argv);

    void dump() const;

private:
    void help() const;
};




#endif
