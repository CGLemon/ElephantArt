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
        LOGGING << "Raw NN output:" << std::endl
                    << std::fixed << std::setprecision(2)
                    << std::setw(11) << "stm eval:" << ' ' << stm_eval * 100.f << "%" << std::endl
                    << std::setw(11) << "winloss:" << ' ' << winloss * 100.f << "%" << std::endl
                    << std::setw(11) << "draw:" << ' ' << nn_eval.draw * 100.f << "%" << std::endl;
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

void Search::play_simulation(Position &currpos, UCTNode *const node,
                             UCTNode *const root_node, SearchResult &search_result) {
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
        UCTNode* next = nullptr;

        if (m_playouts.load() < parameters()->cap_playouts) {
            next = node->prob_select_child();
        } else {
            next = node->uct_select_child(color, node == root_node);
        }

        auto maps = next->get_maps();
        auto move = Decoder::maps2move(maps);
        currpos.do_move_assume_legal(move);
        play_simulation(currpos, next, root_node, search_result);
    }
    if (search_result.valid()) {
        node->update(search_result.nn_evals());
    }
    node->decrement_threads();
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
        for (int i = 0; i < 10; ++i) {
            const auto policy = analysis[i].first;
            const auto maps = analysis[i].second;
            const auto move = Decoder::maps2move(maps);
            out << std::fixed << std::setprecision(2)
                    << "Move "
                    << move.to_string()
                    << " -> Policy: raw "
                    << policy
                    << " | normalize "
                    << policy / acc
                    << std::endl;
        }
        LOGGING << out.str();
    }
    return move;
}

Move Search::get_random_move() {
    const auto maps = m_rootnode->randomize_first_proportionally(1);
    return Decoder::maps2move(maps);
}

Move Search::uct_move() {
    auto info = SearchInformation{};
    auto setting = SearchSetting{};
    think(setting, &info);

    // Wait the threads running finish.
    m_search_group->wait_all();

    auto move = info.best_move;

    if (info.plies < option<int>("random_plies_cnt")) {
        move = info.prob_move;
    }

    return move;
}
