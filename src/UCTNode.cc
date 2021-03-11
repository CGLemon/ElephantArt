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

#include "UCTNode.h"
#include "Board.h"
#include "Random.h"
#include "Utils.h"
#include "config.h"
#include "Decoder.h"
#include "Repetition.h"
#include "ForcedCheckmate.h"

#include <thread>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

UCTNode::UCTNode(std::shared_ptr<UCTNodeData> data) {
    assert(data->parameters != nullptr);
    m_data = data;
    increment_nodes();
}

UCTNode::~UCTNode() {
    assert(get_threads() == 0);
    decrement_nodes();
    release_all_children();
    for (auto i = size_t{0}; i < m_children.size(); ++i) {
        decrement_edges();
    }
}

bool UCTNode::expend_children(Network &network,
                              Position &pos,
                              const float min_psa_ratio,
                              const bool is_root) {
    assert(!pos.gameover(true));

    if (!acquire_expanding()) {
        return false;
    }

    const auto raw_netlist = network.get_output(&pos);

    m_color = pos.get_to_move();
    link_nn_output(raw_netlist, m_color);

    auto inferior_moves = std::vector<Network::PolicyMapsPair>{};
    auto nodelist = std::vector<Network::PolicyMapsPair>{};
    float legal_accumulate = 0.0f;
    float inferior_legal = 0.0f;

    auto movelist = pos.get_movelist();
    const auto kings = pos.get_kings();

    // Probe forced checkmate sequences.
    auto forced = ForcedCheckmate(pos);
    auto ch_move = forced.find_checkmate(movelist);
    if (ch_move.valid()) {
        nodelist.emplace_back(1.0f, Decoder::move2maps(ch_move));
        legal_accumulate = 1.0f;
        movelist.clear();
        set_result(m_color);
    }

    for (const auto &move: movelist) {
        const auto maps = Decoder::move2maps(move);
        const auto policy = raw_netlist.policy[maps];
        if (is_root) {
            auto fork_pos = std::make_shared<Position>(pos);
            fork_pos->do_move_assume_legal(move);
            auto rep = Repetition(*fork_pos);
            auto res = rep.judge();
            if (res == Repetition::UNKNOWN) {
                // It is unknown result. we don't need to consider it if we have
                // other choice. But if not, we will add inferior moves to the
                // node list.
                inferior_legal += policy;
                inferior_moves.emplace_back(policy, maps);
                continue;
            } else if (res == Repetition::LOSE) {
                // If we are lose. Don't need to consider this move.
                continue;
            } else if (res == Repetition::DRAW) {
                // Do nothing.
            }

            if (fork_pos->is_check(Board::swap_color(m_color))) {
                continue;
            }
        }

        if (move.get_to() == kings[Board::swap_color(m_color)]) {
            // We eat opponent's king. Don't need to consider other moves.
            nodelist.clear();
            nodelist.emplace_back(policy, maps);
            legal_accumulate = policy;
            break;
        }

        nodelist.emplace_back(policy, maps);
        legal_accumulate += policy;
    }

    if (nodelist.empty()) {
        if (inferior_moves.empty()) {
            // We are already lose. Pick a random move to the list.
            const auto &move = movelist[0];
            const auto maps = Decoder::move2maps(move);
            const auto policy = raw_netlist.policy[maps];
            nodelist.emplace_back(policy, maps);
        } else {
            legal_accumulate = inferior_legal;
            nodelist = inferior_moves;
        }
    }

    if (legal_accumulate > 0.0f) {
        for (auto &node: nodelist) {
            node.first /= legal_accumulate;
        }
    } else {
        const auto cnt = static_cast<float>(nodelist.size());
        for (auto &node: nodelist) {
            node.first = 1.0f / cnt;
        }
    }

    link_nodelist(nodelist, min_psa_ratio);
    expand_done();
    
    return true;
}

void UCTNode::link_nodelist(std::vector<Network::PolicyMapsPair> &nodelist, float min_psa_ratio) {
    std::stable_sort(std::rbegin(nodelist), std::rend(nodelist));

    const float min_psa = nodelist[0].first * min_psa_ratio;
    for (const auto &node : nodelist) {
        if (node.first < min_psa) {
            break;
        } else {
            auto data = std::make_shared<UCTNodeData>();
            data->maps = node.second;
            data->policy = node.first;
            data->parameters = parameters();
            data->node_status = node_status();
            data->parent = get();
            m_children.emplace_back(std::make_shared<UCTNodePointer>(data));
            increment_edges();
        }
    }
    assert(!m_children.empty());
}

void UCTNode::link_nn_output(const Network::Netresult &raw_netlist,
                             const Types::Color color){

    auto stmeval = raw_netlist.winrate_misc[3];
    auto wl = raw_netlist.winrate_misc[0] - raw_netlist.winrate_misc[2];
    auto draw = raw_netlist.winrate_misc[1];

    stmeval = (stmeval + 1) * 0.5f;
    wl = (wl + 1) * 0.5f;

    if (color == Types::BLACK) {
        stmeval = 1.0f - stmeval;
        wl = 1.0f - wl;
    }
    m_red_stmeval = stmeval;
    m_red_winloss = wl;
    m_draw = draw;
}

void UCTNode::set_result(Types::Color color) {
    if (color == Types::RED) {
        m_red_stmeval = 1;
        m_red_winloss = 1;
        m_draw = 0;
    } else if (color == Types::BLACK) {
        m_red_stmeval = 0;
        m_red_winloss = 0;
        m_draw = 0;
    } else if (color == Types::EMPTY_COLOR) {
        m_red_stmeval = 0.5;
        m_red_winloss = 0.5;
        m_draw = 1;
    }
}

const std::vector<std::shared_ptr<UCTNode::UCTNodePointer>> &UCTNode::get_children() const {
    return m_children;
}

UCTNodeEvals UCTNode::get_node_evals() const {
    auto evals = UCTNodeEvals{};

    evals.red_stmeval = m_red_stmeval;
    evals.red_winloss = m_red_winloss;
    evals.draw = m_draw;

    return evals;
}

int UCTNode::get_maps() const {
    return m_data->maps;
}

float UCTNode::get_policy() const {
    return m_data->policy;
}

int UCTNode::get_visits() const {
    return m_visits.load();
}

float UCTNode::get_nn_stmeval(const Types::Color color) const {
    if (color == Types::RED) {
        return m_red_stmeval;
    }
    return 1.0f - m_red_stmeval;
}

float UCTNode::get_nn_winloss(const Types::Color color) const {
    if (color == Types::RED) {
        return m_red_winloss;
    }
    return 1.0f - m_red_winloss;
}

float UCTNode::get_nn_meaneval(const Types::Color color) const {
    return (get_nn_stmeval(color) + get_nn_winloss(color)) * 0.5f;
}

float UCTNode::get_nn_draw() const {
    return m_draw;
}

float UCTNode::get_accumulated_evals() const {
    return m_accumulated_red_stmevals.load();
}

float UCTNode::get_accumulated_wls() const {
    return m_accumulated_red_wls.load();
}

float UCTNode::get_accumulated_draws() const {
    return m_accumulated_draws.load();
}

Types::Color UCTNode::get_color() const {
    return m_color;
}

UCTNode *UCTNode::get() {
    return this;
}

float UCTNode::get_eval_variance(const float default_var, const int visits) const {
    return visits > 1 ? m_squared_eval_diff.load() / (visits - 1) : default_var;
}

float UCTNode::get_eval_lcb(const Types::Color color) const {
    // LCB issues : https://github.com/leela-zero/leela-zero/pull/2290
    // Lower confidence bound of winrate.
    const auto visits = get_visits();
    if (visits < 2) {
        // Return large negative value if not enough visits.
        return get_policy() - 1e6f;
    }

    const auto mean = get_meaneval(color, false);
    const auto variance = get_eval_variance(1.0f, visits);
    const auto stddev = std::sqrt(variance / float(visits));
    const auto z = Utils::cached_t_quantile(visits - 1);
    
    return mean - z * stddev;
}

int UCTNode::get_threads() const {
    return m_loading_threads.load();
}

int UCTNode::get_virtual_loss() const {
    const auto threads = get_threads();
    const auto virtual_loss = threads * VIRTUAL_LOSS_COUNT;
    return virtual_loss;
}

float UCTNode::get_stmeval(const Types::Color color,
                           const bool use_virtual_loss) const {
    auto virtual_loss = get_virtual_loss();
    auto visits = get_visits();

    if (use_virtual_loss) {
        // If this node is seaching, punish this node.
        visits += virtual_loss;
    }

    assert(visits >= 0);
    auto accumulated_evals = get_accumulated_evals();
    if (color == Types::BLACK && use_virtual_loss) {
        accumulated_evals += static_cast<float>(virtual_loss);
    }
    auto eval = accumulated_evals / static_cast<float>(visits);

    if (color == Types::RED) {
        return eval;
    }
    return 1.0f - eval;
}

float UCTNode::get_winloss(const Types::Color color,
                           const bool use_virtual_loss) const {
    auto virtual_loss = get_virtual_loss();
    auto visits = get_visits();

    if (use_virtual_loss) {
        // If this node is seaching, punish this node.
        visits += virtual_loss;
    }

    assert(visits >= 0);
    auto accumulated_wls = get_accumulated_wls();
    if (color == Types::BLACK && use_virtual_loss) {
        accumulated_wls += static_cast<float>(virtual_loss);
    }
    auto wl = accumulated_wls / static_cast<float>(visits);

    if (color == Types::RED) {
        return wl;
    }
    return 1.0f - wl;
}

float UCTNode::get_meaneval(const Types::Color color,
                            const bool use_virtual_loss) const {
    return (get_winloss(color, use_virtual_loss) + get_stmeval(color, use_virtual_loss)) * 0.5f;
}

float UCTNode::get_draw() const {
    auto visits = get_visits();
    auto accumulated_draws = get_accumulated_draws();
    auto draw = accumulated_draws / static_cast<float>(visits);
    return draw;
}

UCTNode *UCTNode::get_child(const int maps) {
    wait_expanded();
    assert(has_children());

    std::shared_ptr<UCTNodePointer> res = nullptr;

    for (const auto &child : m_children) { 
        const int child_maps = child->data()->maps;
        if (maps == child_maps) {
            res = child;
            break;
        }
    }

    assert(res != nullptr);
    inflate(res);
    return res->get();
}

std::vector<std::pair<float, int>> UCTNode::get_lcb_list(const Types::Color color) {
    wait_expanded();
    assert(has_children());
    assert(color == m_color);
  
    auto list = std::vector<std::pair<float, int>>{};
    inflate_all_children();

    for (const auto & child : m_children) {
        const auto node = child->get();
        const auto visits = node->get_visits();
        const auto maps = node->get_maps();
        const auto lcb = node->get_eval_lcb(color);
        if (visits > 0) {
            list.emplace_back(lcb, maps);
        }
    }

    std::stable_sort(std::rbegin(list), std::rend(list));
    return list;
}

std::vector<std::pair<float, int>> UCTNode::get_winrate_list(const Types::Color color) {
    wait_expanded();
    assert(has_children());
    assert(color == m_color);

    auto list = std::vector<std::pair<float, int>>{};
    inflate_all_children();

    for (const auto &child : m_children) {
        const auto node = child->get();
        const auto visits = node->get_visits();
        const auto maps = node->get_maps();
        const auto winrate = node->get_meaneval(color, false);
        if (visits > 0) {
            list.emplace_back(winrate, maps);
        }
    }

    std::stable_sort(std::rbegin(list), std::rend(list));
    return list;
}

UCTNode *UCTNode::uct_select_child(const Types::Color color,
                                   const bool is_root) {
    wait_expanded();
    assert(has_children());

    int parentvisits = 0;
    float total_visited_policy = 0.0f;
    for (const auto &child : m_children) {
        const auto node = child->get();
        if (!node) {
            continue;
        }    
        if (node->is_valid()) {
            const auto visits = node->get_visits();
            parentvisits += visits;
            if (visits > 0) {
                total_visited_policy += node->get_policy();
            }
        }
    }

    const auto fpu_reduction_factor = is_root ? parameters()->fpu_root_reduction : parameters()->fpu_reduction;
    const auto cpuct_init           = is_root ? parameters()->cpuct_root_init : parameters()->cpuct_init;
    const auto cpuct_base           = is_root ? parameters()->cpuct_root_base : parameters()->cpuct_base;
    const auto draw_factor          = is_root ? parameters()->draw_root_factor : parameters()->draw_factor;

    const float cpuct = cpuct_init + std::log((float(parentvisits) + cpuct_base + 1) / cpuct_base);
    const float numerator = std::sqrt(float(parentvisits));
    const float fpu_reduction = fpu_reduction_factor * std::sqrt(total_visited_policy);
    const float fpu_value = get_nn_meaneval(color) - fpu_reduction;

    std::shared_ptr<UCTNodePointer> best_node = nullptr;
    float best_value = std::numeric_limits<float>::lowest();

    for (const auto &child : m_children) {
        // Check the node is pointer or not.
        // If not, we can not get most data from child.
        const auto node = child->get();
        const bool is_pointer = node == nullptr ? false : true;

        // If the node was pruned. Skip this time,
        if (is_pointer && !node->is_active()) {
            continue;
        }

        float q_value = fpu_value;
        if (is_pointer) {
            if (node->is_expending()) {
                q_value = -1.0f - fpu_reduction;
            } else if (node->get_visits() > 0) {
                const float eval = node->get_meaneval(color);
                const float draw_value = node->get_draw() * draw_factor;
                q_value = eval + draw_value;
            }
        }

        float denom = 1.0f;
        if (is_pointer) {
            denom += node->get_visits();
        }

        const float psa = child->data()->policy;
        const float puct = cpuct * psa * (numerator / denom);
        const float value = q_value + puct;
        assert(value > std::numeric_limits<float>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = child;
        }
    }

    inflate(best_node);
    return best_node->get();
}

void UCTNode::apply_evals(std::shared_ptr<UCTNodeEvals> evals) {
    m_red_stmeval = evals->red_stmeval;
    m_red_winloss = evals->red_winloss;
    m_draw = evals->draw;
}

void UCTNode::update(std::shared_ptr<UCTNodeEvals> evals) {
    const float eval = 0.5f * (evals->red_stmeval + evals->red_winloss);
    const float old_stmeval = m_accumulated_red_stmevals.load();
    const float old_winloss = m_accumulated_red_wls.load();
    const float old_visits = m_visits.load();
    const float old_eval = 0.5f * (old_stmeval + old_winloss);
    const float old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    const float new_delta = eval - (old_eval + eval) / (old_visits + 1);

    // Welford's online algorithm for calculating variance.
    const float delta = old_delta * new_delta;

    m_visits.fetch_add(1);
    Utils::atomic_add(m_squared_eval_diff, delta);
    Utils::atomic_add(m_accumulated_red_stmevals, evals->red_stmeval);
    Utils::atomic_add(m_accumulated_red_wls, evals->red_winloss);
    Utils::atomic_add(m_accumulated_draws, evals->draw);
}

std::vector<float> UCTNode::apply_dirichlet_noise(const float epsilon, const float alpha) {
    auto child_cnt = m_children.size();
    auto dirichlet_buffer = std::vector<float>(child_cnt);
    auto gamma = std::gamma_distribution<float>(alpha, 1.0f);

    std::generate(std::begin(dirichlet_buffer), std::end(dirichlet_buffer),
                      [&gamma] () { return gamma(Random<random_t::XoroShiro128Plus>::get_Rng()); });

    auto sample_sum =
        std::accumulate(std::begin(dirichlet_buffer), std::end(dirichlet_buffer), 0.0f);

    // If the noise vector sums to 0 or a denormal, then don't try to
    // normalize.
    if (sample_sum < std::numeric_limits<float>::min()) {
        std::fill(std::begin(dirichlet_buffer), std::end(dirichlet_buffer), 0.0f);
        return dirichlet_buffer;
    }

    for (auto &v : dirichlet_buffer) {
        v /= sample_sum;
    }

    child_cnt = 0;
    // Be Sure all node are expended.
    inflate_all_children();
    for (const auto &child : m_children) {
        auto node = child->get();
        auto policy = node->get_policy();
        auto eta_a = dirichlet_buffer[child_cnt++];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
        node->set_policy(policy);
    }
    return dirichlet_buffer;
}

UCTNodeEvals UCTNode::prepare_root_node(Network &network,
                                        Position &position,
                                        std::vector<float> &dirichlet) {
    const auto noise = parameters()->dirichlet_noise;
    const auto is_root = true;
    const auto success = expend_children(network, position, 0.0f, is_root);
    const auto had_childen = has_children();
    dirichlet.clear();
    assert(success && had_childen);

    if (success && had_childen) {
        inflate_all_children();
        if (noise) {
            const auto legal_move = m_children.size();
            const auto epsilon = parameters()->dirichlet_epsilon;
            const auto factor = parameters()->dirichlet_factor;
            const auto init = parameters()->dirichlet_init;
            const auto alpha = init * factor / static_cast<float>(legal_move);
            dirichlet = apply_dirichlet_noise(epsilon, alpha);
        }
    }

    return get_node_evals();
}

int UCTNode::get_best_move() {
    wait_expanded();
    assert(has_children());

    auto lcblist = get_lcb_list(m_color);
    float best_value = std::numeric_limits<float>::lowest();
    int best_move = -1;

    for (auto &lcb : lcblist) {
        const auto lcb_value = lcb.first;
        const auto maps = lcb.second;
        if (lcb_value > best_value) {
            best_value = lcb_value;
            best_move = maps;
        }
    }

    if (lcblist.empty() && has_children()) {
        best_move = m_children[0]->get()->get_maps();
    }

    assert(best_move != -1);
    return best_move;
}

int UCTNode::randomize_first_proportionally(float random_temp) {
    auto select_maps = -1;
    auto accum = float{0.0f};
    auto accum_vector = std::vector<std::pair<float, int>>{};

    for (const auto &child : m_children) {
        auto node = child->get();
        const auto visits = node->get_visits();
        const auto maps = node->get_maps();
        if (visits > parameters()->random_min_visits) {
           accum += std::pow((float)visits, (1.0 / random_temp));
           accum_vector.emplace_back(std::pair<float, int>(accum, maps));
        }
    }

    auto distribution = std::uniform_real_distribution<float>{0.0, accum};
    auto pick = distribution(Random<random_t::XoroShiro128Plus>::get_Rng());
    auto size = accum_vector.size();

    for (auto idx = size_t{0}; idx < size; ++idx) {
        if (pick < accum_vector[idx].first) {
            select_maps = accum_vector[idx].second;
            break;
        }
    }

    return select_maps;
}

void UCTNode::increment_threads() {
    m_loading_threads.fetch_add(1);
}

void UCTNode::decrement_threads() {
    m_loading_threads.fetch_sub(1);
}

void UCTNode::increment_nodes() {
    node_status()->nodes.fetch_add(1);
}

void UCTNode::decrement_nodes() {
    node_status()->nodes.fetch_sub(1); 
}

void UCTNode::increment_edges() {
    node_status()->edges.fetch_add(1); 
}

void UCTNode::decrement_edges() {
    node_status()->edges.fetch_sub(1); 
}


void UCTNode::set_active(const bool active) {
    if (is_valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

void UCTNode::invalinode() {
    if (is_valid()) {
        m_status = INVALID;
    }
}

bool UCTNode::has_children() const { 
    return m_color != Types::INVALID_COLOR; 
}

bool UCTNode::expandable() const {
    return m_expand_state.load() == ExpandState::INITIAL;
}

bool UCTNode::is_expending() const {
    return m_expand_state.load() == ExpandState::EXPANDING;
}

bool UCTNode::is_expended() const {
    return m_expand_state.load() == ExpandState::EXPANDED;
}

bool UCTNode::is_pruned() const {
    return m_status.load() == PRUNED;
}

bool UCTNode::is_active() const {
    return m_status.load() == ACTIVE;
}

bool UCTNode::is_valid() const {
    return m_status.load() != INVALID;
}

std::shared_ptr<SearchParameters> UCTNode::parameters() const {
    return m_data->parameters;
}

std::shared_ptr<UCTNodeStats> UCTNode::node_status() const {
    return m_data->node_status;
}

void UCTNode::set_policy(const float p) {
    m_data->policy = p;
}

void UCTNode::inflate_all_children() {
    for (const auto &child : m_children) {
        inflate(child);
    }
}

void UCTNode::release_all_children() {
    for (const auto &child : m_children) {
         release(child);
    }
}

void UCTNode::inflate(std::shared_ptr<UCTNodePointer> child) {
    auto success = child->inflate();
    if (success) {
        decrement_edges();
        increment_nodes();
    }
}

void UCTNode::release(std::shared_ptr<UCTNodePointer> child) {
    auto success = child->release();
    if (success) {
        decrement_nodes();
        increment_edges();
    }
}

bool UCTNode::acquire_expanding() {
    auto expected = ExpandState::INITIAL;
    auto newval = ExpandState::EXPANDING;
    return m_expand_state.compare_exchange_strong(expected, newval);
}

void UCTNode::expand_done() {
    auto v = m_expand_state.exchange(ExpandState::EXPANDED);
    assert(v == ExpandState::EXPANDING);
}

void UCTNode::expand_cancel() {
    auto v = m_expand_state.exchange(ExpandState::INITIAL);
    assert(v == ExpandState::EXPANDING);
}

void UCTNode::wait_expanded() const {
    while (true) {
        auto v = m_expand_state.load();
        if (v == ExpandState::EXPANDED) {
            break;
        }
    }
}

size_t UCT_Information::get_memory_used(UCTNode *node) {
    const auto status = node->node_status();
    const auto nodes = status->nodes.load();
    const auto edges = status->edges.load();
    const auto node_mem = sizeof(UCTNode) + sizeof(NodePointer<UCTNode, UCTNodeData>);
    const auto edge_mem = sizeof(NodePointer<UCTNode, UCTNodeData>);
    return nodes * node_mem + edges * edge_mem;
}

void UCT_Information::dump_tree_stats(UCTNode *node) {
    const auto mem = static_cast<double>(get_memory_used(node)) / (1024.f * 1024.f);
    const auto status = node->node_status();
    const auto nodes = status->nodes.load();
    const auto edges = status->edges.load();

    Utils::printf<Utils::STATIC>("Tree Status: \n");
    Utils::printf<Utils::STATIC>("  nodes: %d, edges: %d, tree memory used: %.2f MiB\n", nodes, edges, mem);
}

void UCT_Information::dump_stats(UCTNode *node, Position &position, int cut_off) {
    const auto color = position.get_to_move();
    const auto lcblist = node->get_lcb_list(color);
    const auto parentvisits = static_cast<float>(node->get_visits());
    assert(color == node->get_color());

    Utils::printf<Utils::STATIC>("Search List:\n"); 
    Utils::printf<Utils::STATIC>("Root -> %7d (WL: %5.2f%%) (V: %5.2f%%) (D: %5.2f%%)\n",
                                     node->get_visits(),
                                     node->get_winloss(color, false) * 100.f,
                                     node->get_stmeval(color, false) * 100.f,
                                     node->get_draw() * 100.f);

    int push = 0;
    for (auto &lcb : lcblist) {
        const auto lcb_value = lcb.first > 0.0f ? lcb.first : 0.0f;
        const auto maps = lcb.second;
    
        auto child = node->get_child(maps);
        const auto visits = child->get_visits();
        const auto pobability = child->get_policy();
        assert(visits != 0);

        const auto wl_eval = child->get_winloss(color, false);
        const auto stm_eval = child->get_stmeval(color, false);
        const auto draw = child->get_draw();
        const auto move = Decoder::maps2move(maps);
        const auto pv_string = move.to_string() + " " + pv_to_srting(child);
        const auto visit_ratio = static_cast<float>(visits) / (parentvisits - 1); // One is root visit.
        Utils::printf<Utils::STATIC>("  %4s -> %7d (WL: %5.2f%%) (V: %5.2f%%) (LCB: %5.2f%%) (D: %5.2f%%) (P: %5.2f%%) (N: %5.2f%%) ", 
                                         move.to_string().c_str(),
                                         visits,
                                         wl_eval * 100.f,    // win loss eval
                                         stm_eval * 100.f,   // side to move eval
                                         lcb_value * 100.f,  // LCB eval
                                         draw * 100.f,       // draw probability
                                         pobability * 100.f, // move probability
                                         visit_ratio * 100.f);
        Utils::printf<Utils::STATIC>("PV: %s\n", pv_string.c_str());

        push++;
        if (push == cut_off) {
            Utils::printf<Utils::STATIC>("     ...remain %d selections\n", (int)lcblist.size() - cut_off);
            break;
        }
    }
    dump_tree_stats(node);
}

std::string UCT_Information::pv_to_srting(UCTNode *node) {
    auto pvlist = std::vector<int>{};
    auto *next = node;
    while (next->has_children()) {
        const auto maps = next->get_best_move();
        pvlist.emplace_back(maps);
        next = next->get_child(maps);
    }
  
    auto res = std::string{};
    for (const auto &maps : pvlist) {
        const auto move = Decoder::maps2move(maps);
        res += move.to_string();
        res += " ";
    }
    return res;
}
