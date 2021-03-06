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

#include "config.h"
#include "Zobrist.h"
#include "Board.h"
#include "Decoder.h"
#include "Utils.h"
#include "Search.h"

#include <limits>
#include <string>

std::unordered_map<std::string, Utils::Option> options_map;

#define OPTIONS_EXPASSION(T)                        \
template<>                                          \
T option<T>(std::string name) {                     \
    return options_map.find(name)->second.get<T>(); \
}                                                   \

OPTIONS_EXPASSION(std::string)
OPTIONS_EXPASSION(const char*)
OPTIONS_EXPASSION(bool)
OPTIONS_EXPASSION(int)
OPTIONS_EXPASSION(float)
OPTIONS_EXPASSION(char)

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
OPTIONS_SET_EXPASSION(const char*)
OPTIONS_SET_EXPASSION(bool)
OPTIONS_SET_EXPASSION(int)
OPTIONS_SET_EXPASSION(float)
OPTIONS_SET_EXPASSION(char)

#undef OPTIONS_SET_EXPASSION

void init_options_map() {
    options_map["name"] << Utils::Option::setoption(PROGRAM);
    options_map["version"] << Utils::Option::setoption(VERSION);

    options_map["mode"] << Utils::Option::setoption("ascii");
    options_map["help"] << Utils::Option::setoption(false);

    options_map["gpu"] << Utils::Option::setoption(0);
    options_map["batchsize"] << Utils::Option::setoption(1, 256, 1);
    options_map["threads"] << Utils::Option::setoption(1, 256, 1);

    options_map["quiet_verbose"] << Utils::Option::setoption(false);
    options_map["stats_verbose"] << Utils::Option::setoption(false);
    options_map["analysis_verbose"] << Utils::Option::setoption(false);
    options_map["ucci_response"] << Utils::Option::setoption(true);
    options_map["log_file"] << Utils::Option::setoption(NO_LOG_FILE_NAME);

    options_map["num_games"] << Utils::Option::setoption(1, 32, 1);

    options_map["softmax_pol_temp"] << Utils::Option::setoption(1.0f);
    options_map["softmax_wdl_temp"] << Utils::Option::setoption(1.0f);
    options_map["cache_moves"] << Utils::Option::setoption(20);
    options_map["weights_file"] << Utils::Option::setoption(NO_WEIGHT_FILE_NAME);
    options_map["float_precision"] << Utils::Option::setoption(5);
    options_map["winograd"] << Utils::Option::setoption(false);
    options_map["min_cutoff"] << Utils::Option::setoption(1);

    options_map["ponder"] << Utils::Option::setoption(false);
    options_map["playouts"] << Utils::Option::setoption(Search::MAX_PLAYOUTS);
    options_map["visits"] << Utils::Option::setoption(Search::MAX_PLAYOUTS);
    options_map["fpu_root_reduction"] << Utils::Option::setoption(0.25f);
    options_map["fpu_reduction"] << Utils::Option::setoption(0.25f);
    options_map["cpuct_init"] << Utils::Option::setoption(2.5f);
    options_map["cpuct_root_init"] << Utils::Option::setoption(2.5f);
    options_map["cpuct_base"] << Utils::Option::setoption(19652.f);
    options_map["cpuct_root_base"] << Utils::Option::setoption(19652.f);
    options_map["draw_factor"] << Utils::Option::setoption(0.f);
    options_map["draw_root_factor"] << Utils::Option::setoption(0.f);

    options_map["collect"] << Utils::Option::setoption(false);
    options_map["collection_buffer_size"] << Utils::Option::setoption(1000, 10000, 0);
    options_map["random_min_visits"] << Utils::Option::setoption(1);
    options_map["random_move_cnt"] << Utils::Option::setoption(0);

    options_map["dirichlet_noise"] << Utils::Option::setoption(false);
    options_map["dirichlet_epsilon"] << Utils::Option::setoption(0.25f);
    options_map["dirichlet_init"] << Utils::Option::setoption(0.3f);
    options_map["dirichlet_factor"] << Utils::Option::setoption(60.f);

    options_map["gpu_waittime"] << Utils::Option::setoption(10);
    options_map["use_gpu"] << Utils::Option::setoption(false);

    options_map["black_pawn_en"] << Utils::Option::setoption('p');
    options_map["black_horse_en"] << Utils::Option::setoption('n');
    options_map["black_cannon_en"] << Utils::Option::setoption('c');
    options_map["black_rook_en"] << Utils::Option::setoption('r');
    options_map["black_elephant_en"] << Utils::Option::setoption('b');
    options_map["black_advisor_en"] << Utils::Option::setoption('a');
    options_map["black_king_en"] << Utils::Option::setoption('k');

    options_map["red_pawn_en"] << Utils::Option::setoption('P');
    options_map["red_horse_en"] << Utils::Option::setoption('N');
    options_map["red_cannon_en"] << Utils::Option::setoption('C');
    options_map["red_rook_en"] << Utils::Option::setoption('R');
    options_map["red_elephant_en"] << Utils::Option::setoption('B');
    options_map["red_advisor_en"] << Utils::Option::setoption('A');
    options_map["red_king_en"] << Utils::Option::setoption('K');

    options_map["using_traditional_chinese"] << Utils::Option::setoption(false);
    options_map["black_pawn_ch"] << Utils::Option::setoption("卒");
    options_map["black_horse_ch"] << Utils::Option::setoption("馬");
    options_map["black_cannon_ch"] << Utils::Option::setoption("砲");
    options_map["black_rook_ch"] << Utils::Option::setoption("車");
    options_map["black_elephant_ch"] << Utils::Option::setoption("象");
    options_map["black_advisor_ch"] << Utils::Option::setoption("士");
    options_map["black_king_ch"] << Utils::Option::setoption("將");

    options_map["red_pawn_ch"] << Utils::Option::setoption("兵");
    options_map["red_horse_ch"] << Utils::Option::setoption("傌");
    options_map["red_cannon_ch"] << Utils::Option::setoption("炮");
    options_map["red_rook_ch"] << Utils::Option::setoption("俥");
    options_map["red_elephant_ch"] << Utils::Option::setoption("像");
    options_map["red_advisor_ch"] << Utils::Option::setoption("仕");
    options_map["red_king_ch"] << Utils::Option::setoption("帥");
}

void init_basic_parameters() {
    Zobrist::init_zobrist();
    Decoder::initialize();
    Board::pre_initialize();
}

ArgsParser::ArgsParser(int argc, char** argv) {
    auto parser = Utils::CommandParser(argc, argv);
    const auto is_parameter = [](const std::string &param) -> bool {
        if (param.empty()) {
            return false;
        }
        return param[0] != '-';
    };

    const auto error_commands = [is_parameter](Utils::CommandParser & parser) -> bool {
        const auto cnt = parser.get_count();
        if (cnt == 0) {
            return false;
        }
        int t = 1;
        Utils::printf<Utils::STATIC>("Command(s) Error!\n  The parameter(s)\n");
        for (auto i = size_t{0}; i < cnt; ++i) {
            const auto command = parser.get_command(i)->str;
            if (!is_parameter(command)) {
                if (t != 1) { Utils::printf<Utils::STATIC>("\n"); }
                Utils::printf<Utils::STATIC>("    %d. %s", t++, command.c_str());
            }
        }
        Utils::printf<Utils::STATIC>("\n  are not understood.\n");
        return true;
    };

    init_options_map();

    const auto name = parser.remove_command(0);
    (void) name;

    if (const auto res = parser.find({"--help", "-h"})) {
        set_option("help", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find({"--chinese", "-ch"})) {
        set_option("using_traditional_chinese", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find({"--quiet", "-q"})) {
        set_option("quiet_verbose", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find("--stats_verbose")) {
        set_option("stats_verbose", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find("--analysis_verbose")) {
        set_option("analysis_verbose", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find("--debug_mode")) {
        set_option("quiet_verbose", false);
        set_option("stats_verbose", true);
        set_option("analysis_verbose", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find("--collect")) {
        set_option("collect", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find_next({"--logfile", "-l"})) {
        if (is_parameter(res->str)) {
            set_option("log_file", res->get<std::string>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--mode", "-m"})) {
        if (is_parameter(res->str)) {
            if (res->str == "ascii"
                    || res->str == "ucci"
                    || res->str == "selfplay") {
                set_option("mode", res->get<std::string>());
                parser.remove_slice(res->idx-1, res->idx+1);
            }
        }
    }

    if (const auto res = parser.find_next({"--playouts", "-p"})) {
        if (is_parameter(res->str)) {
            set_option("playouts", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--visits", "-v"})) {
        if (is_parameter(res->str)) {
            set_option("visits", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--threads", "-t"})) {
        if (is_parameter(res->str)) {
            set_option("threads", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--weight", "-w"})) {
        if (is_parameter(res->str)) {
            set_option("weights_file", res->get<std::string>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--floatprecision")) {
        if (is_parameter(res->str)) {
            set_option("float_precision", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--batchsize" , "-b"})) {
        if (is_parameter(res->str)) {
            set_option("batchsize", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--gpu", "-g"})) {
        if (is_parameter(res->str)) {
            set_option("gpu", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black_pawn")) {
        if (is_parameter(res->str)) {
            set_option<char>("black_pawn_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black_horse")) {
        if (is_parameter(res->str)) {
            set_option<char>("black_horse_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black_cannon")) {
        if (is_parameter(res->str)) {
            set_option<char>("black_cannon_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black_rook")) {
        if (is_parameter(res->str)) {
            set_option<char>("black_rook_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black_elephant")) {
        if (is_parameter(res->str)) {
            set_option<char>("black_elephant_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black_advisor")) {
        if (is_parameter(res->str)) {
            set_option<char>("black_advisor_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black_king")) {
        if (is_parameter(res->str)) {
            set_option<char>("black_king_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red_pawn")) {
        if (is_parameter(res->str)) {
            set_option<char>("red_pawn_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red_horse")) {
        if (is_parameter(res->str)) {
            set_option<char>("red_horse_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red_cannon")) {
        if (is_parameter(res->str)) {
            set_option<char>("red_cannon_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red_rook")) {
        if (is_parameter(res->str)) {
            set_option<char>("red_rook_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red_elephant")) {
        if (is_parameter(res->str)) {
            set_option<char>("red_elephant_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red_advisor")) {
        if (is_parameter(res->str)) {
            set_option<char>("red_advisor_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red_king")) {
        if (is_parameter(res->str)) {
            set_option<char>("red_king_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }
    
#ifdef USE_CUDA
    set_option("use_gpu", true);
#endif
    
    if (error_commands(parser)) {
        help();
    }
    if (option<std::string>("mode") == "ucci") {
        set_option("quiet_verbose", true);
    }
}

void ArgsParser::help() const {
    Utils::printf<Utils::SYNC>("Arguments:\n");
    Utils::printf<Utils::SYNC>("  --help, -h\n");
    Utils::printf<Utils::SYNC>("  --chinese, -ch\n");
    Utils::printf<Utils::SYNC>("  --mode, -m [ascii/ucci]\n");
    Utils::printf<Utils::SYNC>("  --playouts, -p <integer>\n");
    Utils::printf<Utils::SYNC>("  --threads, -t <integer>\n");
    Utils::printf<Utils::SYNC>("  --weights, -w <weight file name>\n");
    exit(-1);
}

void ArgsParser::dump() const {
    if (option<bool>("help")) {
        help();
    }
}
