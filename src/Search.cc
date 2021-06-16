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

#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "Board.h"
#include "Search.h"
#include "UCTNode.h"
#include "Random.h"
#include "Model.h"
#include "Decoder.h"
#include "config.h"

Search::Search(Position &position, Network &network, Train &train) : 
    m_position(position),  m_network(network), m_train(train) {

    m_parameters = std::make_shared<SearchParameters>();

    const auto t = m_parameters->threads;
    m_searchpool.initialize(t);
    m_search_group = std::make_unique<ThreadGroup<void>>(m_searchpool);

    m_maxplayouts = m_parameters->playouts;
    m_maxvisits = m_parameters->visits;
}

std::shared_ptr<SearchParameters> Search::parameters() const {
    return m_parameters;
}

Search::~Search() {
    clear_nodes();
    m_search_group->wait_all();
}

bool Search::is_running() {
    return m_running.load();
}

void Search::set_running(bool running) {
    m_running.store(running);
}

void Search::set_playouts(int playouts) {
    m_playouts.store(playouts);
}

bool Search::stop_thinking(int elapsed, int limittime) const {
   return elapsed > limittime ||
              m_playouts.load() >= m_maxplayouts ||
              m_rootnode->get_visits() >= m_maxvisits;
}

std::pair<Move, Move> Search::get_best_move() const {
    const auto maps = m_rootnode->get_best_move();
    auto best_move = Decoder::maps2move(maps);

    auto child = m_rootnode->get_child(maps);
    auto ponder_move = Move{};
    if (child->has_children()) {
        const auto ponder_maps = child->get_best_move();
        ponder_move = Decoder::maps2move(ponder_maps);
    }
    return std::make_pair(best_move, ponder_move);
}

std::string Search::get_draw_resign(Types::Color color, bool draw) const {
    const auto winrate = m_rootnode->get_meaneval(color, false);
    if (winrate < parameters()->resign_threshold) {
        if (draw) {
            return std::string{"draw"};
        } else {
            return std::string{"resign"};
        }
    }

    const auto draw_prob = m_rootnode->get_draw();
    if (draw_prob > parameters()->draw_threshold) {
        return std::string{"draw"};
    }
    return std::string{};
}

void Search::increment_threads() {
    m_running_threads.fetch_add(1);
}

void Search::decrement_threads() {
    m_running_threads.fetch_sub(1);
}

void Search::think(SearchSetting setting, SearchInformation *info) {
    if (is_running()) {
        return;
    }

    m_search_group->wait_all();
    m_rootposition = m_position;
    if (m_rootposition.gameover(true)) {
        return;
    }

    const auto uct_worker = [this]() -> void {
        // Waiting, until main thread searching.
        while (m_running_threads.load() < 1 && is_running()) {
            std::this_thread::yield();
        }
        increment_threads();
        while(is_running()) {
            auto currpos = std::make_unique<Position>(m_rootposition);
            auto result = SearchResult{};
            play_simulation(*currpos, m_rootnode, m_rootnode, result);
            if (result.valid()) {
                increment_playouts();
            }
        };
        decrement_threads();
    };

    const auto main_uct_worker = [this, set = setting, info]() -> void {
        auto keep_running = true;
        auto maxdepth = 0;
        const auto limitnodes = set.nodes;
        const auto limitdepth = set.depth;
        auto controller = TimeControl(set.milliseconds,
                                      set.movestogo,
                                      set.increment);
        controller.set_plies(m_rootposition.get_gameply());

        auto timer = Utils::Timer{};
        auto limittime = std::numeric_limits<int>::max();
        auto ponder_lock = false;

        prepare_uct();
        {
            // Stop it if preparing uct is time out.  
            if (!set.ponder) {
                limittime = controller.get_limittime();
                ponder_lock = true;
            }
            const auto elapsed = timer.get_duration_milliseconds();
            set_running(!stop_thinking(elapsed, limittime));
        }

        increment_threads();
        while(is_running()) {
            auto currpos = std::make_unique<Position>(m_rootposition);
            auto result = SearchResult{};

            play_simulation(*currpos, m_rootnode, m_rootnode, result);
            if (result.valid()) {
                increment_playouts();
            }

            const auto color = m_rootposition.get_to_move();
            const auto score = (m_rootnode->get_meaneval(color, false) - 0.5f) * 200.0f;
            const auto nodes = m_nodestats->nodes.load() + m_nodestats->edges.load();
            const auto elapsed = timer.get_duration_milliseconds();
            controller.set_score(int(score));
            {
                std::lock_guard<std::mutex> lock(m_thinking_mtx);
                if (!set.ponder && !ponder_lock) {
                    // The ponderhit is valid now. Start to time clock.
                    limittime = controller.get_limittime() + elapsed;
                    ponder_lock = true;
                }
            }

            const auto pv_depth = (int)UCT_Information::get_pvlist(m_rootnode).size();

            keep_running &= (!stop_thinking(elapsed, limittime));
            keep_running &= (!(limitnodes < nodes));
            keep_running &= (!(limitdepth < pv_depth));
            keep_running &= is_running();
            set_running(keep_running);

            if (option<bool>("ucci_response") &&
                   (pv_depth > maxdepth || !keep_running) &&
                    m_playouts.load() >= parameters()->cap_playouts) {
                if (keep_running) {
                    maxdepth = pv_depth;
                }
                const auto pv = UCT_Information::get_pvsrting(m_rootnode);
                LOGGING << "info"
                            << ' ' << "depth" << ' ' << maxdepth
                            << ' ' << "time"  << ' ' << elapsed
                            << ' ' << "nodes" << ' ' << nodes
                            << ' ' << "score" << ' ' << int(score)
                            << ' ' << "pv"    << ' ' << pv
                            << std::endl;
            }
        }

        decrement_threads();

        // Waiting, until all threads finish searching.
        while (m_running_threads.load() != 0) {
            std::this_thread::yield();
        }

        m_train.gather_probabilities(*m_rootnode, m_rootposition);

        const auto color = m_rootposition.get_to_move();
        const auto elapsed = timer.get_duration();
        const auto moves = get_best_move();
        const auto bestmove = moves.first;
        const auto pondermove = moves.second;
        const auto prob_move = get_random_move();
        const auto draw_resign = get_draw_resign(color, set.draw);

        if (option<bool>("ucci_response")) {
            LOGGING << "bestmove" << ' ' << bestmove.to_string();
            if (pondermove.valid()) {
                LOGGING << ' ' << "ponder" << ' ' << pondermove.to_string();
            }
            if (!draw_resign.empty()) {
                LOGGING << ' ' << draw_resign;
            }
            LOGGING << std::endl;
        }

        if (info) {
            info->best_move = bestmove;
            info->prob_move = prob_move;
            info->seconds = elapsed;
            info->depth = maxdepth;
            info->plies =  m_rootposition.get_gameply();
        }
        if (option<bool>("analysis_verbose")) {
            LOGGING << UCT_Information::get_stats_string(m_rootnode, m_rootposition);
            LOGGING << "Speed:" << std::endl
                        << "  " << elapsed                   << ' ' << "second(s)" << std::endl
                        << "  " << m_playouts.load()         << ' ' << "playout(s)" << std::endl
                        << "  " << m_playouts.load()/elapsed << ' ' << "p/s" << std::endl;
        }
        clear_nodes();
    };
    set_running(true);
    m_search_group->add_task(main_uct_worker);
    m_search_group->add_tasks(m_parameters->threads-1, uct_worker);
}

void Search::interrupt() {
    set_running(false);
    m_search_group->wait_all();
}

void Search::ponderhit(bool draw) {
    std::lock_guard<std::mutex> lock(m_thinking_mtx);
    m_setting.ponder = false;
    m_setting.draw = draw;
}
