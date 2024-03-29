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
#include <iomanip>

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

    const auto raw_netlist = network.get_output(&pos, Network::RANDOM_SYMMETRY);

    m_color = pos.get_to_move();
    link_nn_output(raw_netlist, m_color);

    auto inferior_moves = std::vector<Network::PolicyMapsPair>{};
    auto nodelist = std::vector<Network::PolicyMapsPair>{};
    float legal_accumulate = 0.0f;
    float inferior_legal = 0.0f;

    auto movelist = pos.get_movelist();
    const auto kings = pos.get_kings();

    // Probe forced checkmate sequences.
    const auto maxdepth = is_root ? parameters()->forced_checkmate_root_depth :
                                        parameters()->forced_checkmate_depth;

    auto forced_move = pos.get_forced_checkmate_move(maxdepth);
    if (forced_move.valid()) {
        nodelist.emplace_back(1.0f, Decoder::move2maps(forced_move));

        set_result(get_color());
        link_nodelist(nodelist, min_psa_ratio);

        auto child = m_children[0];
        inflate(child);
        child->get()->make_terminated(get_color());
        expand_done();
        return true;
    }

    for (const auto &move : movelist) {
        const auto maps = Decoder::move2maps(move);
        const auto policy = raw_netlist.policy[maps];
        if (is_root) {
            auto fork_pos = std::make_shared<Position>(pos);
            fork_pos->do_move_assume_legal(move);

            const auto res = fork_pos->get_threefold_repetitions_result();
            if (res == Position::Repetition::UNKNOWN) {
                // It is unknown result. we don't need to consider it if we have
                // other choice. But if not, we will add inferior moves to the
                // node list.
                inferior_legal += policy;
                inferior_moves.emplace_back(policy, maps);
                continue;
            } else if (res == Position::Repetition::LOSE) {
                // If we are lose. Don't need to consider this move.
                continue;
            } else if (res == Position::Repetition::DRAW) {
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
        for (auto &node : nodelist) {
            node.first /= legal_accumulate;
        }
    } else {
        const auto cnt = static_cast<float>(nodelist.size());
        for (auto &node : nodelist) {
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
    auto wl = raw_netlist.winrate_misc[0] - raw_netlist.winrate_misc[2];
    auto draw = raw_netlist.winrate_misc[1];

    wl = (wl + 1) * 0.5f;

    if (color == Types::BLACK) {
        wl = 1.0f - wl;
    }

    m_red_winloss = wl;
    m_draw = draw;
}

void UCTNode::set_result(Types::Color color) {
    if (color == Types::RED) {
        m_red_winloss = 1;
        m_draw = 0;
    } else if (color == Types::BLACK) {
        m_red_winloss = 0;
        m_draw = 0;
    } else if (color == Types::EMPTY_COLOR) {
        m_red_winloss = 0.5;
        m_draw = 1;
    }
}

void UCTNode::policy_target_pruning() {
    wait_expanded();
    assert(has_children());
    inflate_all_children();

    auto buffer = std::vector<std::pair<int, int>>{};

    int parentvisits = 0;
    int most_visits_maps = -1;
    int most_visits = 0;

    for (const auto &child : m_children) {
        const auto node = child->get();

        if (!node->is_active()) {
            continue;
        }

        const auto visits = node->get_visits();
        const auto maps = node->get_maps();
        parentvisits += visits;
        buffer.emplace_back(visits, maps);

        if (most_visits < visits) {
            most_visits = visits;
            most_visits_maps = maps;
        }
    }

    assert(!buffer.empty());

    const auto forced_policy_factor = parameters()->forced_policy_factor;
    for (const auto &x : buffer) {
        const auto visits = x.first;
        const auto maps = x.second;
        auto node = get_child(maps);

        auto forced_playouts = std::sqrt(forced_policy_factor *
                                             node->get_policy() *
                                             float(parentvisits));
        auto new_visits = std::max(visits - int(forced_playouts), 0);
        node->set_visits(new_visits);
    }

    while (true) {
        auto node = uct_select_child(get_color(), false);
        if (node->get_maps() == most_visits_maps) {
            break;
        }
        node->set_active(false);
    }

    for (const auto &x : buffer) {
        const auto visits = x.first;
        const auto maps = x.second;
        get_child(maps)->set_visits(visits);
    }
}

const std::vector<std::shared_ptr<UCTNode::UCTNodePointer>> &UCTNode::get_children() const {
    return m_children;
}

UCTNodeEvals UCTNode::get_node_evals() const {
    auto evals = UCTNodeEvals{};

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

float UCTNode::get_nn_winloss(const Types::Color color) const {
    if (color == Types::RED) {
        return m_red_winloss;
    }
    return 1.0f - m_red_winloss;
}

float UCTNode::get_nn_draw() const {
    return m_draw;
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
    // LCB issues: https://github.com/leela-zero/leela-zero/pull/2290
    // Lower confidence bound of winrate.
    const auto visits = get_visits();
    if (visits < 2) {
        // Return large negative value if not enough visits.
        return get_policy() - 1e6f;
    }

    const auto mean = get_winloss(color, false);
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

float UCTNode::get_draw() const {
    return get_accumulated_draws() / static_cast<float>(get_visits());
}

UCTNode *UCTNode::get_child(const int maps) {
    wait_expanded();
    assert(has_children());

    std::shared_ptr<UCTNodePointer> res = nullptr;

    for (const auto &child: m_children) { 
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
        const auto winrate = node->get_winloss(color, false);
        if (visits > 0) {
            list.emplace_back(winrate, maps);
        }
    }

    std::stable_sort(std::rbegin(list), std::rend(list));
    return list;
}

float UCTNode::get_uct_policy(std::shared_ptr<UCTNodePointer> child, bool noise) const {
    auto policy = child->data()->policy;
    if (noise) {
        const auto maps = child->data()->maps;
        const auto epsilon = parameters()->dirichlet_epsilon;
        const auto eta_a = parameters()->dirichlet_buffer[maps];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
    }
    return policy;
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
    const auto forced_policy_factor = is_root ? parameters()->forced_policy_factor : 0.0f;
    const auto noise                = is_root ? parameters()->dirichlet_noise : false;

    const float cpuct = cpuct_init + std::log((float(parentvisits) + cpuct_base + 1) / cpuct_base);
    const float numerator = std::sqrt(float(parentvisits));
    const float fpu_reduction = fpu_reduction_factor * std::sqrt(total_visited_policy);
    const float fpu_value = get_nn_winloss(color) - fpu_reduction;

    std::shared_ptr<UCTNodePointer> best_node = nullptr;
    float best_value = std::numeric_limits<float>::lowest();

    for (const auto &child : m_children) {
        // Check the node is pointer or not.
        // If not, we can not get most data from child.
        const auto node = child->get();
        const bool is_pointer = node == nullptr ? false : true;

        // The node was pruned. Skip this time.
        if (is_pointer && !node->is_active()) {
            continue;
        }

        float q_value = fpu_value;
        if (is_pointer) {
            if (node->is_expending()) {
                q_value = -1.0f - fpu_reduction;
            } else if (node->get_visits() > 0) {
                const float eval = node->get_winloss(color, true);
                const float draw_value = node->get_draw() * draw_factor;
                q_value = eval + draw_value;
            }
        }

        float denom = 1.0f;
        float bonus = 0.0f;
        if (is_pointer) {
            denom += node->get_visits();
            auto forced_playouts = std::sqrt(forced_policy_factor *
                                                 node->get_policy() *
                                                 float(parentvisits));
            bonus += (int) (forced_playouts - denom + 1.0f);
            bonus *= 10;
            bonus = std::max(bonus, 0.0f);
        }

        const float psa = get_uct_policy(child, noise);
        const float puct = cpuct * psa * (numerator / denom);
        const float value = q_value + puct + bonus;
        assert(value > std::numeric_limits<float>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = child;
        }
    }

    inflate(best_node);
    return best_node->get();
}

UCTNode *UCTNode::prob_select_child() {
    wait_expanded();
    assert(has_children());

    std::shared_ptr<UCTNodePointer> best_node = nullptr;
    float best_prob = std::numeric_limits<float>::lowest();

    for (const auto &child : m_children) {
        const auto node = child->get();
        const bool is_pointer = node == nullptr ? false : true;

        auto prob = child->data()->policy;

        // The node was pruned. Skip this time.
        if (is_pointer && !node->is_active()) {
            continue;
        }

        // The node was expending.
        if (is_pointer && node->is_expending()) {
            prob = -1.0f + prob;
        }

        if (prob > best_prob) {
            best_prob = prob;
            best_node = child;
        }
    }

    inflate(best_node);
    return best_node->get();
}

void UCTNode::apply_evals(std::shared_ptr<UCTNodeEvals> evals) {
    m_red_winloss = evals->red_winloss;
    m_draw = evals->draw;
}

void UCTNode::update(std::shared_ptr<UCTNodeEvals> evals) {
    const float eval = evals->red_winloss;
    const float old_eval = m_accumulated_red_wls.load();
    const float old_visits = m_visits.load();

    const float old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    const float new_delta = eval - (old_eval + eval) / (old_visits + 1);

    // Welford's online algorithm for calculating variance.
    const float delta = old_delta * new_delta;

    m_visits.fetch_add(1);
    Utils::atomic_add(m_squared_eval_diff, delta);
    Utils::atomic_add(m_accumulated_red_wls, evals->red_winloss);
    Utils::atomic_add(m_accumulated_draws, evals->draw);
}

void UCTNode::apply_dirichlet_noise(const float alpha) {
    auto child_cnt = m_children.size();
    auto buffer = std::vector<float>(child_cnt);
    auto gamma = std::gamma_distribution<float>(alpha, 1.0f);

    std::generate(std::begin(buffer), std::end(buffer),
                      [&gamma] () { return gamma(Random<random_t::XoroShiro128Plus>::get_Rng()); });

    auto sample_sum =
        std::accumulate(std::begin(buffer), std::end(buffer), 0.0f);

    // Clear dirichlet buffer.
    parameters()->dirichlet_buffer.fill(0.0f);

    // If the noise vector sums to 0 or a denormal, then don't try to
    // normalize.
    if (sample_sum < std::numeric_limits<float>::min()) {
        return;
    }

    for (auto &v : buffer) {
        v /= sample_sum;
    }

    child_cnt = 0;

    for (const auto &child : m_children) {
        const auto maps = child->data()->maps;
        parameters()->dirichlet_buffer[maps] = buffer[child_cnt++];
    }
}

UCTNodeEvals UCTNode::prepare_root_node(Network &network,
                                        Position &position) {
    const auto noise = parameters()->dirichlet_noise;
    const auto is_root = true;
    const auto success = expend_children(network, position, 0.0f, is_root);
    const auto had_childen = has_children();
    assert(success && had_childen);

    if (success && had_childen) {
        inflate_all_children();
        if (noise) {
            const auto legal_move = m_children.size();
            const auto factor = parameters()->dirichlet_factor;
            const auto init = parameters()->dirichlet_init;
            const auto alpha = init * factor / static_cast<float>(legal_move);
            apply_dirichlet_noise(alpha);
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

void UCTNode::make_terminated(Types::Color color) {
    set_result(color);
    m_terminated = true;
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

void UCTNode::set_visits(int v) {
    m_visits.store(v);
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

bool UCTNode::is_terminated() const {
    return m_terminated;
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
    if (child->inflate()) {
        decrement_edges();
        increment_nodes();
    }
}

void UCTNode::release(std::shared_ptr<UCTNodePointer> child) {
    if (child->release()) {
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
#ifdef NDEBUG
    (void) v;
#endif
    assert(v == ExpandState::EXPANDING);
}

void UCTNode::expand_cancel() {
    auto v = m_expand_state.exchange(ExpandState::INITIAL);
#ifdef NDEBUG
    (void) v;
#endif
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

std::string UCT_Information::get_stats_string(UCTNode *node, Position &position) {
    auto out = std::ostringstream{};
    const auto color = position.get_to_move();
    const auto lcblist = node->get_lcb_list(color);
    const auto parentvisits = static_cast<float>(node->get_visits());
    assert(color == node->get_color());


    const auto space = 7;
    out << "Search List:" << std::endl;
    out << std::setw(6) << "move"
            << std::setw(10) << "visits"
            << std::setw(space) << "WL(%)"
            << std::setw(space) << "V(%)"
            << std::setw(space) << "LCB(%)"
            << std::setw(space) << "D(%)"
            << std::setw(space) << "P(%)"
            << std::setw(space) << "N(%)"
            << std::endl;

    for (auto &lcb : lcblist) {
        const auto lcb_value = lcb.first > 0.0f ? lcb.first : 0.0f;
        const auto maps = lcb.second;

        auto child = node->get_child(maps);
        const auto visits = child->get_visits();
        const auto pobability = child->get_policy();
        assert(visits != 0);

        const auto wl_eval = child->get_winloss(color, false);
        const auto draw = child->get_draw();
        const auto move = Decoder::maps2move(maps);
        const auto pv_string = move.to_string() + ' ' + get_pvsrting(child);
        const auto visit_ratio = static_cast<float>(visits) / (parentvisits - 1); // One is root visit.
        out << std::fixed << std::setprecision(2)
                << std::setw(6) << move.to_string()
                << std::setw(10) << visits
                << std::setw(space) << wl_eval * 100.f     // win loss eval
                << std::setw(space) << lcb_value * 100.f   // LCB eval
                << std::setw(space) << draw * 100.f        // draw probability
                << std::setw(space) << pobability * 100.f  // move probability
                << std::setw(space) << visit_ratio * 100.f
                << std::setw(6) << "| PV:" << ' ' << pv_string
                << std::endl;
    }

    out << get_memory_string(node);

    return out.str();
}

std::string UCT_Information::get_memory_string(UCTNode *node) {
    const auto mem = static_cast<double>(get_memory_used(node)) / (1024.f * 1024.f);
    const auto status = node->node_status();
    const auto nodes = status->nodes.load();
    const auto edges = status->edges.load();

    auto out = std::ostringstream{};

    out << "Tree Status:" << std::endl
            << std::setw(9) << "nodes:" << ' ' << nodes  << std::endl
            << std::setw(9) << "edges:" << ' ' << edges  << std::endl
            << std::setw(9) << "memory:" << ' ' << mem << ' ' << "(MiB)" << std::endl;

    return out.str();
}

std::string UCT_Information::get_pvsrting(UCTNode *node) {
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

std::vector<int> UCT_Information::get_pvlist(UCTNode *node) {
    auto pvlist = std::vector<int>{};
    auto *next = node;
    while (next->has_children()) {
        const auto maps = next->get_best_move();
        pvlist.emplace_back(maps);
        next = next->get_child(maps);
    }
    return pvlist;
}
