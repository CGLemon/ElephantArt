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

#include "Engine.h"
#include "config.h"
#include "Model.h"
#include "Decoder.h"
#include "Utils.h"
#include "PGNParser.h"

#include <iomanip>
#include <sstream>

void Engine::initialize() {
    const auto games = (size_t)option<int>("num_games");

    m_positions.clear();
    m_train_group.clear();
    m_search_group.clear();

    while (m_positions.size() < games) {
        m_positions.emplace_back(std::make_shared<Position>());
    }
    
    int tag = 0;
    for (auto &p : m_positions) {
        p->init_game(tag++);
    }
    
    if (m_network == nullptr) {
        m_network = std::make_unique<Network>();
        m_network->initialize(option<std::string>("weights_file"));
    }

    while (m_train_group.size() < games) {
        m_train_group.emplace_back(std::make_shared<Train>());
    }

    while (m_search_group.size() < games) {
        const auto s = m_search_group.size();
        m_search_group.emplace_back(std::make_shared<Search>(
                                        *get_position(s),
                                        *m_network,
                                        *get_train(s)));
    }
}

int Engine::clamp(const int g) const {
    if (g < 0 || g >= option<int>("num_games")) {
        return DEFUALT_POSITION;
    }
    return g;
}

std::shared_ptr<Position> Engine::get_position(const int g) const {
    const auto adj_g = clamp(g);
    return m_positions[adj_g];
}

std::shared_ptr<Search> Engine::get_search(const int g) const {
    const auto adj_g = clamp(g);
    return m_search_group[adj_g];
}

std::shared_ptr<Train> Engine::get_train(const int g) const {
    const auto adj_g = clamp(g);
    return m_train_group[adj_g];
}

void Engine::reset_game(const int g) {
    assert(g >= 0 || g <= option<int>("num_games"));
    get_position(g)->init_game(g);
}


void Engine::display(const int g) const {
     get_position(g)->display();
}

Engine::Response Engine::gather_movelist(const int g) {
    auto rep = std::ostringstream{};
    const auto movelist = get_position(g)->get_movelist();

    for (const auto &m: movelist) {
        rep << m.to_string() << " ";
    }

    return rep.str();
}

Engine::Response Engine::fen(std::string fen, const int g) {
    auto rep = std::ostringstream{};
    auto success = get_position(g)->fen(fen);
    if (!success) {
        rep << "Illegal FEN format";
    }

    return rep.str();
}

Engine::Response Engine::do_textmove(std::string move, const int g) {
    auto rep = std::ostringstream{};
    auto success = get_position(g)->do_textmove(move);
    if (!success) {
        rep << "Illegal move";
    }
    return rep.str();
}

Engine::Response Engine::position(std::string pos, const int g) {
    auto rep = std::ostringstream{};
    auto parser = Utils::CommandParser(pos);

    auto startpos = false;
    auto fen_idx = -1;
    auto moves_idx = -1;

    auto fen = std::string{};
    auto moves = std::string{};
    if (const auto res = parser.find("startpos")) {
        fen_idx = res->idx-1;
        startpos = true;
    } else if (const auto res = parser.find("fen")) {
        fen_idx = res->idx;
    }

    if (const auto res = parser.find("moves")) {
        moves_idx = res->idx;
    }
    
    if (fen_idx != -1 || startpos) {
        if (moves_idx != -1) {
            fen = parser.get_slice(fen_idx+1, moves_idx)->str;
            moves = parser.get_commands(moves_idx+1)->str;
        } else {
            fen = parser.get_commands(fen_idx+1)->str;
        }
    }

    if (!get_position(g)->position(fen, moves)) {
        rep << "Illegal position";
    }
    return rep.str();
}

Engine::Response Engine::raw_nn(const int g) {
    auto rep = std::ostringstream{};
    auto pres = option<int>("float_precision");
    auto max_index = 0;
    auto timer = Utils::Timer{};
    auto &p = *get_position(g);
    auto nnout = m_network->get_output(&p);
    auto microsecond = timer.get_duration_microseconds();
    for (int p = 0; p < POLICYMAP; ++p) {
        rep << "map probabilities: " << p+1 << std::endl;
        for (int y = 0; y < Board::HEIGHT; ++y) {
            for (int x = 0; x < Board::WIDTH; ++x) {
                const auto idx = Board::get_index(x, y);
                rep << std::fixed
                    << std::setprecision(pres)
                    << nnout.policy[idx + p * Board::INTERSECTIONS]
                    << " ";
                if (nnout.policy[idx + p * Board::INTERSECTIONS] > nnout.policy[max_index]) {
                    max_index = idx + p * Board::INTERSECTIONS;
                }
            }
            rep << std::endl;
        }
        rep << std::endl;
    }
    rep << "max " << max_index << " probability: " << std::setprecision(pres) << nnout.policy[max_index] << std::endl;
    auto m = Decoder::maps2move(max_index);
    rep << "max move: " << m.to_string() << std::endl << std::endl;

    rep << "wdl probabilities ( win / draw / loss ): " << std::endl;
    for (int v = 0; v < 3; ++v) {
        rep << nnout.winrate_misc[v] << " ";
    }
    rep << std::endl << std::endl;
    rep << "stm winrate: " << std::endl;
    rep << nnout.winrate_misc[3];
    rep << std::endl << std::endl;

    rep << "run time ";
    rep << microsecond;
    rep << " microsecond(s)" << std::endl;

    return rep.str();
}

Engine::Response Engine::input_planes(const int g) {
    auto rep = std::ostringstream{};
    const auto &p = *get_position(g);
    const auto input_planes = Model::gather_planes(&p, false);
    for (int p = 0; p < INPUT_CHANNELS; ++p) {
        rep << "planes: " << p+1 << std::endl;
        for (int y = 0; y < Board::HEIGHT; ++y) {
            for (int x = 0; x < Board::WIDTH; ++x) {
                const auto idx = Board::get_index(x, y);
                rep << std::setw(5)
                    << input_planes[idx + p * Board::INTERSECTIONS]
                    << " ";
            }
            rep << std::endl;
        }
        rep << std::endl;
    }
    
    const auto input_features = Model::gather_features(&p);
    rep << "features" << std::endl;
    for (int f = 0; f < INPUT_FEATURES; ++f) {
        rep << std::setw(5) << input_features[f] << std::endl;
    }
    rep << std::endl;
    return rep.str();
}

Engine::Response Engine::history_board(const int g) {
    auto rep = std::ostringstream{};
    const auto p = get_position(g);
    rep << p->history_board();
    return rep.str();
}

Engine::Response Engine::rand_move(const int g) {
    auto rep = std::ostringstream{};
    const auto p = get_position(g); 
    const auto s = get_search(g);

    const auto move = s->random_move();
    const auto success = p->do_move(move);
    assert(success);

    return rep.str();
}

Engine::Response Engine::nn_direct_move(const int g) {
    auto rep = std::ostringstream{};
    const auto p = get_position(g); 
    const auto s = get_search(g);

    const auto move = s->nn_direct_move();
    const auto success = p->do_move(move);
    assert(success);

    return rep.str();
}

Engine::Response Engine::uct_move(const int g) {
    auto rep = std::ostringstream{};
    const auto p = get_position(g);
    if (p->gameover(true)) {
        return rep.str();
    }

    const auto s = get_search(g);

    const auto move = s->uct_move();
    const auto success = p->do_move(move);
    assert(success);

    return rep.str();
}

Engine::Response Engine::dump_collection(std::string filename, const int g) {
    auto rep = std::ostringstream{};
    auto t = get_train(g);
    if (filename == "NO_FILE_NAME") {
        t->data_stream(rep);
    } else {
        t->save_data(filename);
        t->clear_buffer();
    }
    return rep.str();
}

Engine::Response Engine::selfplay(const int g) {
    auto rep = std::ostringstream{};
    auto p = get_position(g); 
    auto s = get_search(g);
    auto t = get_train(g);
    while (!p->gameover(true)) {
        display();
        const auto move = s->uct_move();
        const auto success = p->do_move(move);
        assert(success);
    }
    const auto winner = p->get_winner(false);
    assert(winner != Types::INVALID_COLOR);
    if (winner == Types::BLACK) {
        rep << "Black is winner";
    } else if (winner == Types::RED) {
        rep << "Red is winner";
    } else if (winner == Types::EMPTY_COLOR) {
        rep << "Draw";
    }
    t->gather_winner(winner);
    return rep.str();
}

Engine::Response Engine::get_maps() {
    return Decoder::get_mapstring();
}

Engine::Response Engine::printf_pgn(std::string filename, const int g) {
    auto rep = std::ostringstream{};
    auto parser = PGNParser{};
    auto &p = *get_position(g);
    if (filename != "NO_FILE_NAME") {
        parser.savepgn(filename, p);
    } else {
        parser.pgn_stream(rep, p);
    }
    return rep.str();
}

Engine::Response Engine::supervised(std::string filename, std::string outname, const int g) {
    auto rep = std::ostringstream{};
    auto t = get_train(g);
    t->supervised(filename, outname);
    return rep.str();
}

Engine::Response Engine::load_pgn(std::string filename, const int g) {
    auto rep = std::ostringstream{};
    auto parser = PGNParser{};
    auto &p = *get_position(g);
    parser.loadpgn(filename, p);
    return rep.str();
}

Engine::Response Engine::think(SearchSetting setting, const int g) {
    auto s = get_search(g);
    s->think(setting, nullptr);
    return std::string{};
}

Engine::Response Engine::interrupt(const int g) {
    auto s = get_search(g);
    s->interrupt();
    return std::string{};
}

Engine::Response Engine::ponderhit(const bool draw, const int g) {
    auto s = get_search(g);
    s->ponderhit(draw);
    return std::string{};
}

Engine::Response Engine::newgame(const int g) {
    auto p = get_position(g); 
    m_network->clear_cache();
    p->init_game(clamp(g));
    return std::string{};
}

Engine::Response Engine::setoption(std::string key, std::string val, const int g) {
    if (key == "cachesize") {
        const auto mem = std::stoi(val);
        m_network->set_cache_memory(mem);
    } else if (key == "usemillisec") {
        if (val == "true") {
            set_option(key, true);
        } else if (val == "false") {
            set_option(key, false);
        }
    } else if (key == "playouts") {
        auto s = get_search(g);
        s->set_playouts(std::stoi(val));
    } else if (key == "cpuct-init") {
        auto s = get_search(g);
        s->parameters()->cpuct_init = std::stoi(val);
    } else if (key == "cpuct-root-init") {
        auto s = get_search(g);
        s->parameters()->cpuct_root_init = std::stoi(val);
    } else if (key == "cpuct-base") {
        auto s = get_search(g);
        s->parameters()->cpuct_base = std::stoi(val);
    } else if (key == "cpuct-root-base") {
        auto s = get_search(g);
        s->parameters()->cpuct_root_base = std::stoi(val);
    }

    return std::string{};
}
