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

#include "config.h"
#include "Zobrist.h"
#include "Board.h"
#include "Decoder.h"
#include "Utils.h"

#include <string>

std::unordered_map<std::string, Utils::Option> options_map;

#define OPTIONS_EXPASSION(T)                        \
template<>                                          \
T option<T>(std::string name) {                     \
    return options_map.find(name)->second.get<T>(); \
}                                                   \

OPTIONS_EXPASSION(std::string)
OPTIONS_EXPASSION(bool)
OPTIONS_EXPASSION(int)
OPTIONS_EXPASSION(float)

#undef OPTIONS_EXPASSION

#define OPTIONS_SET_EXPASSION(T)                     \
template<>                                           \
bool set_option<T>(std::string name, T val) {        \
    auto res = options_map.find(name);               \
    if (res != std::end(options_map)) {              \
        res->second.set<T>(val);                     \
        return true;                                 \
    }                                                \
    return false;                                    \
}

OPTIONS_SET_EXPASSION(std::string)
OPTIONS_SET_EXPASSION(bool)
OPTIONS_SET_EXPASSION(int)
OPTIONS_SET_EXPASSION(float)

#undef OPTIONS_SET_EXPASSION

void init_options_map() {

    options_map["name"] << Utils::Option::setoption(PROGRAM);
    options_map["version"] << Utils::Option::setoption(VERSION);

    options_map["mode"] << Utils::Option::setoption("ascii");
    options_map["help"] << Utils::Option::setoption(false);

    options_map["gpu"] << Utils::Option::setoption(0);
    options_map["batchsize"] << Utils::Option::setoption(1, 256, 1);

    options_map["quiet"] << Utils::Option::setoption(false);
    options_map["num_games"] << Utils::Option::setoption(1, 32, 1);
    options_map["reserve_movelist"] << Utils::Option::setoption(60);

    options_map["softmax_temp"] << Utils::Option::setoption(1.0f);
    options_map["cache_moves"] << Utils::Option::setoption(20);
    options_map["weights_file"] << Utils::Option::setoption("NO_WEIGHT_FILE");
    options_map["float_precision"] << Utils::Option::setoption(5);
    
    options_map["playouts"] << Utils::Option::setoption(1600);
    options_map["fpu_root_reduction"] << Utils::Option::setoption(0.25f);
    options_map["fpu_reduction"] << Utils::Option::setoption(0.25f);
    options_map["logconst"] << Utils::Option::setoption(0.015f);
    options_map["logpuct"] << Utils::Option::setoption(1.7f);
    options_map["cpuct"] << Utils::Option::setoption(0.5f);
    options_map["random_min_visits"] << Utils::Option::setoption(1);
    
    options_map["dirichlet_noise"] << Utils::Option::setoption(false);
    options_map["ponder"]          << Utils::Option::setoption(false);
    options_map["collect"]         << Utils::Option::setoption(false);
}

void init_basic_parameters() {
    Zobrist::init_zobrist();
    Decoder::initialize();
    Board::pre_initialize();
}

ArgsParser::ArgsParser(int argc, char** argv) {

    auto parser = Utils::CommandParser(argc, argv);
    const auto is_parameter = [](const std::string &para) -> bool {
        if (para.empty()) {
            return false;
        }
        return para[0] != '-';
    };

    init_options_map();
    const auto name = parser.remove_command(0);

    if (const auto res = parser.find({"--help", "-h"})) {
        set_option("help", true);
    }

    if (const auto res = parser.find_next({"--mode", "-m"})) {
        if (is_parameter(res->str)) {
            if (res->str == "ascii" || res->str == "ucci") {
                set_option("mode", res->get<std::string>());
            }
        }
    }

    if (const auto res = parser.find_next({"--playouts", "-p"})) {
        if (is_parameter(res->str)) {
            set_option("playouts", res->get<int>());
        }
    }

    if (const auto res = parser.find_next({"--weight", "-w"})) {
        if (is_parameter(res->str)) {
            set_option("weights_file", res->get<std::string>());
        }
    }

    if (const auto res = parser.find_next("--floatprecision")) {
        if (is_parameter(res->str)) {
            set_option("float_precision", res->get<int>());
        }
    }

    if (const auto res = parser.find_next({"--batchsize" , "-b"})) {
        if (is_parameter(res->str)) {
            set_option("batchsize", res->get<int>());
        }
    }

    if (const auto res = parser.find_next("--gpu")) {
        if (is_parameter(res->str)) {
            set_option("gpu", res->get<int>());
        }
    }
}

void ArgsParser::help() const {
    Utils::auto_printf("Argumnet\n");
    Utils::auto_printf(" --help, -h\n");
    Utils::auto_printf(" --mode, -m [ascii/ucci]\n");  
}

void ArgsParser::dump() const {
    if (option<bool>("help")) {
        help();
        exit(-1);
    }
}
