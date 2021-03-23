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
    m_threadGroup = std::make_unique<ThreadGroup<void>>(m_searchpool);

    m_maxplayouts = m_parameters->playouts;
    m_maxvisits = m_parameters->visits;
}

std::shared_ptr<SearchParameters> Search::parameters() const {
    return m_parameters;
}

Search::~Search() {
    clear_nodes();
    m_threadGroup->wait_all();
}

void Search::increment_playouts() {
    m_playouts.fetch_add(1);
}

float Search::get_min_psa_ratio() {
    auto v = m_playouts.load();
    if (v >= MAX_PLAYOUTS) {
        return 1.0f;
    }
    return 0.0f;
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

void Search::play_simulation(Position &currpos, UCTNode *const node,
                             UCTNode *const root_node, SearchResult &search_result, int &depth) {
    node->increment_threads();
    if (node->expandable()) {
        if (currpos.gameover(true)) {
            search_result.from_gameover(currpos);
            node->apply_evals(search_result.nn_evals());
        } else if (node->is_terminated()) {
            search_result.from_nn_evals(node->get_node_evals());
        } else {
            const bool has_children = node->has_children();
            const bool success = node->expend_children(m_network,
                                                       currpos,
                                                       get_min_psa_ratio());
            if (!has_children && success) {
                search_result.from_nn_evals(node->get_node_evals());
            }
        }
    }
    if (node->has_children() && !search_result.valid()) {
        auto color = currpos.get_to_move();
        auto next = node->uct_select_child(color, node == root_node);
        auto maps = next->get_maps();
        auto move = Decoder::maps2move(maps);
        currpos.do_move_assume_legal(move);
        play_simulation(currpos, next, root_node, search_result, depth);
        ++depth;
    }
    if (search_result.valid()) {
        node->update(search_result.nn_evals());
    }
    node->decrement_threads();
}

Move Search::uct_move() {
    auto info = SearchInformation{};
    auto setting = SearchSetting{};
    think(setting, &info);
    // Wait the thread running finish.
    m_threadGroup->wait_all();

    return info.move;
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

void Search::prepare_uct() {
    auto data = std::make_shared<UCTNodeData>();
    m_nodestats = std::make_shared<UCTNodeStats>();
    data->parameters = m_parameters;
    data->node_status = m_nodestats;
    m_rootnode = new UCTNode(data);

    set_playouts(0);
    set_running(true);

    // TODO: According to "Accelerating Self-Play Learning in Go",
    //       implement policy target pruning to improve probabilities.
    auto dirichlet = std::vector<float>{};

    m_rootnode->prepare_root_node(m_network, m_rootposition, dirichlet);
    const auto nn_eval = m_rootnode->get_node_evals();
    m_rootnode->update(std::make_shared<UCTNodeEvals>(nn_eval));

    const auto color = m_rootposition.get_to_move();
    const auto stm_eval = color == Types::RED ? nn_eval.red_stmeval : 1 - nn_eval.red_stmeval;
    const auto winloss = color == Types::RED ? nn_eval.red_winloss : 1 - nn_eval.red_winloss;
    if (option<bool>("analysis_verbose")) {
        Utils::printf<Utils::STATIC>("Raw NN output\n");
        Utils::printf<Utils::STATIC>("  stm eval: %.2f%\n", stm_eval * 100.f);
        Utils::printf<Utils::STATIC>("  winloss: %.2f%\n", winloss * 100.f);
        Utils::printf<Utils::STATIC>("  draw probability: %.2f%\n", nn_eval.draw * 100.f);
    }
}

void Search::clear_nodes() {
    if (m_rootnode) {
        delete m_rootnode;
        m_rootnode = nullptr;
    }
    if (m_nodestats) {
        assert(m_nodestats->nodes.load() == 0);
        assert(m_nodestats->edges.load() == 0);
        m_nodestats.reset();
        m_nodestats = nullptr;
    }
}

Move Search::nn_direct_move() {
    m_rootposition = m_position;
    auto analysis = std::vector<std::pair<float, int>>();
    auto acc = 0.0f;
    auto eval = m_network.get_output(&m_rootposition);
    for (int m = 0; m < POLICYMAP * Board::INTERSECTIONS; ++m) {
        if (!Decoder::maps_valid(m)) {
            continue;
        }

        if (!m_rootposition.is_legal(Decoder::maps2move(m))) {
            continue;
        }
        analysis.emplace_back(eval.policy[m], m);
        acc += eval.policy[m];
    }

    std::stable_sort(std::rbegin(analysis), std::rend(analysis));

    auto move = Decoder::maps2move(std::begin(analysis)->second);
    if (option<bool>("analysis_verbose")) {
        auto out = std::ostringstream{};
        auto prec = option<int>("float_precision");
        for (int i = 0; i < 10; ++i) {
            const auto policy = analysis[i].first;
            const auto maps = analysis[i].second;
            const auto move = Decoder::maps2move(maps);
            out << "Move "
                << move.to_string()
                << " -> Policy: raw "
                << std::fixed
                << std::setprecision(prec)
                << policy
                << " | normalize "
                << std::fixed
                << std::setprecision(prec)
                << policy / acc
                << std::endl;
        }
        Utils::printf<Utils::STATIC>(out);
    }
    return move;
}

Move Search::random_move() {
    m_rootposition = m_position;

    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    int maps = -1;

    while (true) {
        const auto randmaps = rng.randfix<POLICYMAP * Board::INTERSECTIONS>();

        if (!Decoder::maps_valid(randmaps)) {
            continue;
        }

        if (m_rootposition.is_legal(Decoder::maps2move(randmaps))) {
            maps = randmaps;
            break;
        }
    }

    return Decoder::maps2move(maps);
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

    m_threadGroup->wait_all();
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
            auto depth = 0;
            auto currpos = std::make_unique<Position>(m_rootposition);
            auto result = SearchResult{};
            play_simulation(*currpos, m_rootnode, m_rootnode, result, depth);
            if (result.valid()) {
                increment_playouts();
            }
        };
        decrement_threads();
    };

    const auto main_worker = [this, set = setting, info]() -> void {
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
            auto depth = 0;
            auto currpos = std::make_unique<Position>(m_rootposition);
            auto result = SearchResult{};

            play_simulation(*currpos, m_rootnode, m_rootnode, result, depth);
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

            keep_running &= (!stop_thinking(elapsed, limittime));
            keep_running &= (!(limitnodes < nodes));
            keep_running &= (!(limitdepth < depth));
            keep_running &= is_running();
            set_running(keep_running);

            if (option<bool>("ucci_response") && (depth > maxdepth || !keep_running)) {
                if (keep_running) {
                    maxdepth = depth;
                }
                const auto pv = UCT_Information::pv_to_srting(m_rootnode);
                Utils::printf<Utils::SYNC>("info depth %d time %d nodes %d score %d pv %s\n",
                                               maxdepth, elapsed, nodes, int(score), pv.c_str());
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

        const auto draw_resign = get_draw_resign(color, set.draw);

        if (option<bool>("ucci_response")) {
            Utils::printf<Utils::SYNC>("bestmove %s", bestmove.to_string().c_str());
            if (pondermove.valid()) {
                Utils::printf<Utils::SYNC>(" ponder %s", pondermove.to_string().c_str());
            }
            if (draw_resign != "") {
                Utils::printf<Utils::SYNC>(" %s",draw_resign.c_str());
            }
            Utils::printf<Utils::SYNC>("\n");
        }
        if (info) {
            info->move = bestmove;
            info->seconds = elapsed;
            info->depth = maxdepth;
        }
        if (option<bool>("analysis_verbose")) {
            UCT_Information::dump_stats(m_rootnode, m_rootposition);
            Utils::printf<Utils::STATIC>("Speed:\n");
            Utils::printf<Utils::STATIC>("  %.4f second(s), %d playout(s), %.2f p/s\n",
                                              elapsed, m_playouts.load(), m_playouts.load()/elapsed);
        }
        clear_nodes();
    };
    set_running(true);
    m_threadGroup->add_task(main_worker);
    m_threadGroup->add_tasks(m_parameters->threads-1, uct_worker);
}

void Search::interrupt() {
    set_running(false);
    m_threadGroup->wait_all();
}

void Search::ponderhit(bool draw) {
    std::lock_guard<std::mutex> lock(m_thinking_mtx);
    m_setting.ponder = false;
    m_setting.draw = draw;
}
