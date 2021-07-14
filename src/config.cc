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

    options_map["debug_verbose"] << Utils::Option::setoption(false);
    options_map["analysis_verbose"] << Utils::Option::setoption(false);
    options_map["ucci_response"] << Utils::Option::setoption(true);
    options_map["log_file"] << Utils::Option::setoption(std::string{});

    options_map["sync_games"] << Utils::Option::setoption(1, 256, 1);
    options_map["selfplay_games"] << Utils::Option::setoption(0);
    options_map["selfplay_directory"] << Utils::Option::setoption(std::string{});

    options_map["usemillisec"] << Utils::Option::setoption(false);
    options_map["cache_size"] << Utils::Option::setoption(500);
    options_map["cache_playouts"] << Utils::Option::setoption(0);

    options_map["softmax_pol_temp"] << Utils::Option::setoption(1.0f);
    options_map["softmax_wdl_temp"] << Utils::Option::setoption(1.0f);
    options_map["weights_file"] << Utils::Option::setoption(std::string{});
    options_map["winograd"] << Utils::Option::setoption(false);
    options_map["min_cutoff"] << Utils::Option::setoption(1);

    options_map["draw_threshold"] << Utils::Option::setoption(0.9f, 1.f, 0.f);
    options_map["resign_threshold"] << Utils::Option::setoption(0.1f, 1.f, 0.f);
    options_map["playouts"] << Utils::Option::setoption(Search::MAX_PLAYOUTS);
    options_map["cap_playouts"] << Utils::Option::setoption(0);
    options_map["visits"] << Utils::Option::setoption(Search::MAX_PLAYOUTS);
    options_map["forced_policy_factor"] << Utils::Option::setoption(0.f);
    options_map["fpu_reduction"] << Utils::Option::setoption(0.25f);
    options_map["fpu_root_reduction"] << Utils::Option::setoption(0.25f);
    options_map["cpuct_init"] << Utils::Option::setoption(2.5f);
    options_map["cpuct_root_init"] << Utils::Option::setoption(2.5f);
    options_map["cpuct_base"] << Utils::Option::setoption(19652.f);
    options_map["cpuct_root_base"] << Utils::Option::setoption(19652.f);
    options_map["draw_factor"] << Utils::Option::setoption(0.f);
    options_map["draw_root_factor"] << Utils::Option::setoption(0.f);
    options_map["forced_checkmate_depth"] << Utils::Option::setoption(10, 256, 0);
    options_map["forced_checkmate_root_depth"] << Utils::Option::setoption(20, 256, 0);

    options_map["collect"] << Utils::Option::setoption(false);
    options_map["collection_buffer_size"] << Utils::Option::setoption(1000, 10000, 0);
    options_map["random_min_visits"] << Utils::Option::setoption(1);
    options_map["random_plies_cnt"] << Utils::Option::setoption(0);

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

    options_map["using_chinese"] << Utils::Option::setoption(false);
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

    options_map["pgn_format"] << Utils::Option::setoption("iccs");
}

void init_basic_parameters() {
    Zobrist::init_zobrist();
    Board::pre_initialize();
    Decoder::initialize();
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
        int t = 0;
        ERROR << "Command(s) Error:" << std::endl;
        for (auto i = size_t{0}; i < cnt; ++i) {
            const auto command = parser.get_command(i)->str;
            if (!is_parameter(command)) {
                ERROR << ' ' << ++t << '.' << ' ' << command << std::endl;
            }
        }
        ERROR << " are not understood." << std::endl;
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
        set_option("using_chinese", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find("--analysis-verbose")) {
        set_option("analysis_verbose", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find("--collect")) {
        set_option("collect", true);
        parser.remove_command(res->idx);
    }

    if  (const auto res = parser.find("--usemillisec")) {
        set_option("usemillisec", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find_next("--gpu-waittime")) {
        if (is_parameter(res->str)) {
            set_option("gpu_waittime", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--selfplay-games")) {
        if (is_parameter(res->str)) {
            set_option("selfplay_games", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--sync-games")) {
        if (is_parameter(res->str)) {
            set_option("sync_games", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--selfplay-directory")) {
        if (is_parameter(res->str)) {
            set_option("selfplay_directory", res->get<std::string>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
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

    if (const auto res = parser.find_next("--forced-policy-factor")) {
        if (is_parameter(res->str)) {
            set_option("forced_policy_factor", res->get<float>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--fpu-reduction")) {
        if (is_parameter(res->str)) {
            set_option("fpu_reduction", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--fpu-root-reduction")) {
        if (is_parameter(res->str)) {
            set_option("fpu_root_reduction", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--cpuct-init")) {
        if (is_parameter(res->str)) {
            set_option("cpuct-init", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--cpuct-root-init")) {
        if (is_parameter(res->str)) {
            set_option("cpuct_root_init", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--cpuct-base")) {
        if (is_parameter(res->str)) {
            set_option("cpuct_base", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--cpuct-root-base")) {
        if (is_parameter(res->str)) {
            set_option("cpuct_root_base", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--draw-factor")) {
        if (is_parameter(res->str)) {
            set_option("draw_factor", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--draw-root-factor")) {
        if (is_parameter(res->str)) {
            set_option("draw_root_factor", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--playouts", "-p"})) {
        if (is_parameter(res->str)) {
            set_option("playouts", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--cap-playouts"})) {
        if (is_parameter(res->str)) {
            set_option("cap_playouts", res->get<int>());
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

    if (const auto res = parser.find_next("--cache-size")) {
        if (is_parameter(res->str)) {
            set_option("cache_size", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--cache-playouts")) {
        if (is_parameter(res->str)) {
            set_option("cache_playouts", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next({"--weight", "-w"})) {
        if (is_parameter(res->str)) {
            set_option("weights_file", res->get<std::string>());
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

    if (const auto res = parser.find_next("--black-pawn")) {
        if (is_parameter(res->str)) {
            set_option("black_pawn_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black-horse")) {
        if (is_parameter(res->str)) {
            set_option("black_horse_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black-cannon")) {
        if (is_parameter(res->str)) {
            set_option("black_cannon_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black-rook")) {
        if (is_parameter(res->str)) {
            set_option("black_rook_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black-elephant")) {
        if (is_parameter(res->str)) {
            set_option("black_elephant_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black-advisor")) {
        if (is_parameter(res->str)) {
            set_option("black_advisor_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--black-king")) {
        if (is_parameter(res->str)) {
            set_option("black_king_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red-pawn")) {
        if (is_parameter(res->str)) {
            set_option("red_pawn_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red-horse")) {
        if (is_parameter(res->str)) {
            set_option("red_horse_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red-cannon")) {
        if (is_parameter(res->str)) {
            set_option("red_cannon_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red-rook")) {
        if (is_parameter(res->str)) {
            set_option("red_rook_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red-elephant")) {
        if (is_parameter(res->str)) {
            set_option("red_elephant_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red-advisor")) {
        if (is_parameter(res->str)) {
            set_option("red_advisor_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--red-king")) {
        if (is_parameter(res->str)) {
            set_option("red_king_en", res->get<char>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find({"-n", "--noise"})) {
        set_option("dirichlet_noise", true);
        parser.remove_command(res->idx);
    }

    if (const auto res = parser.find_next("--random-plies")) {
        if (is_parameter(res->str)) {
            set_option("random_plies_cnt", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--forced-checkmate-depth")) {
        if (is_parameter(res->str)) {
            set_option("forced_checkmate_depth", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }

    if (const auto res = parser.find_next("--forced-checkmate-root-depth")) {
        if (is_parameter(res->str)) {
            set_option("forced_checkmate_root_depth", res->get<int>());
            parser.remove_slice(res->idx-1, res->idx+1);
        }
    }


    if (const auto res = parser.find_next("--pgn-format")) {
        if (is_parameter(res->str)) {
            if (res->str == "iccs"
                    || res->str == "wxf") {
                set_option("pgn_format", res->get<std::string>());
                parser.remove_slice(res->idx-1, res->idx+1);
            }
        }
    }

#ifdef USE_CUDA
    set_option("use_gpu", true);
#endif

    if (const auto res = parser.find("--debug-mode")) {
        set_option("debug_verbose", true);
        parser.remove_command(res->idx);
    }

    if (error_commands(parser)) {
        help();
    }
}

void ArgsParser::help() const {
    ERROR << "Arguments:" << std::endl
              << "  --help, -h" << std::endl
              << "  --chinese, -ch" << std::endl
              << "  --mode, -m [ascii/ucci]" << std::endl
              << "  --playouts, -p <integer>" << std::endl
              << "  --threads, -t <integer>" << std::endl
              << "  --batchsize, -b <integer>" << std::endl
              << "  --weights, -w <weights file name>" << std::endl
              << "  --log_file, -l <log file name>" << std::endl
              << "  --gpu, -g <integer> " << std::endl
              << "  --analysis-verbose" << std::endl;
    exit(-1);
}

void ArgsParser::dump() const {
    if (option<bool>("help")) {
        help();
    }
}
