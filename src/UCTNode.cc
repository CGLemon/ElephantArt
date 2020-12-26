#include "UCTNode.h"
#include "Board.h"
#include "Random.h"
#include "Utils.h"
#include "config.h"
#include "Decoder.h"

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
    m_parameters = data->parameters;
    m_policy = data->policy;
    m_maps = data->maps;
    m_parent = data->parent;
}

UCTNode::~UCTNode() {
    assert(get_threads() == 0);
}

bool UCTNode::expend_children(Network &network,
                              Position &position,
                              const float min_psa_ratio,
                              const bool is_root) {
    
    if (!acquire_expanding()) {
        return false;
    }
    
    const auto raw_netlist =
        network.get_output(&position, Network::Ensemble::RANDOM_SYMMETRY);
    
    m_color = position.get_to_move();
    link_nn_output(raw_netlist, m_color);
    
    auto nodelist = std::vector<Network::PolicyMapsPair>{};
    float legal_accumulate = 0.0f;

    auto movelist = position.get_movelist();
    for (const auto &move : movelist) {
        if (is_root) { /* Do something... */ }

        const auto maps = Decoder::move2maps(move);
        const auto policy = raw_netlist.policy[maps];
        nodelist.emplace_back(policy, maps);
        legal_accumulate += policy;
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

    std::stable_sort(rbegin(nodelist), rend(nodelist));

    const float min_psa = nodelist[0].first * min_psa_ratio;
    for (const auto &node : nodelist) {
        if (node.first < min_psa) {
            break;
        } else {
            auto data = std::make_shared<UCTNodeData>();
            data->maps = node.second;
            data->policy = node.first;
            data->parameters = parameters();
            data->parent = get();
            m_children.emplace_back(std::make_shared<UCTNodePointer>(data));
        }
    }
    assert(!m_children.empty());
}

void UCTNode::link_nn_output(const Network::Netresult &raw_netlist,
                             const Types::Color color){

    auto stmeval = raw_netlist.winrate_misc[3];
    auto wl = raw_netlist.winrate_misc[0] - raw_netlist.winrate_misc[2];
    auto draw = raw_netlist.winrate_misc[1];

    if (color == Types::BLACK) {        
        stmeval = 1.0f - stmeval;
        wl = 1.0f - wl;
    }
    m_red_stmeval = stmeval;
    m_red_winloss = wl;
    m_draw = draw;
}

UCTNodeEvals UCTNode::get_node_evals() const {

    auto evals = UCTNodeEvals{};

    evals.red_stmeval = m_red_stmeval;
    evals.red_winloss = m_red_winloss;
    evals.draw = m_draw;

    return evals;
}

int UCTNode::get_maps() const {
    return m_maps;
}

float UCTNode::get_policy() const {
    return m_policy;
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

int UCTNode::get_color() const {
    return m_color;
}

UCTNode *UCTNode::get() {
    return this;
}

float UCTNode::get_eval_variance(const float default_var, const int visits) const {
    return visits > 1 ? m_squared_eval_diff / (visits - 1) : default_var;
}

float UCTNode::get_eval_lcb(const Types::Color color) const {
    // LCB issues : https://github.com/leela-zero/leela-zero/pull/2290
    // Lower confidence bound of winrate.
    const auto visits = get_visits();
    if (visits < 2) {
        // Return large negative value if not enough visits.
        return get_policy() - 1e6f;
    }

    const auto mean = (get_stmeval(color, false) + get_winloss(color, false)) / 2.0f;
    const auto stddev = std::sqrt(get_eval_variance(1.0f, visits) / float(visits));
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

float UCTNode::get_draw() const {
    auto visits = get_visits();
    auto accumulated_draws = get_accumulated_draws();
    auto draw = accumulated_draws / static_cast<float>(visits);
    return draw;
}

UCTNode *UCTNode::uct_select_child(const Types::Color color,
                                   const bool is_root) const {
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
    const float fpu_value = float(get_nn_stmeval(color)) - fpu_reduction;

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
                const float stmeval = node->get_stmeval(color);
                const float winloss = node->get_winloss(color);
                const float draw_value = node->get_draw() * draw_factor;
                q_value = (stmeval + winloss) / 2.0f + draw_value;
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

    best_node->inflate();
    return best_node->get();
}

void UCTNode::update(UCTNodeEvals &evals) {

    const float eval = (evals.red_stmeval + evals.red_winloss) / 2.0f;

    const float old_stmeval = m_accumulated_red_stmevals.load();
    const float old_winloss = m_accumulated_red_wls.load();
    const float old_visits = m_visits.load();
    const float old_eval = (old_stmeval + old_winloss) / old_visits;
    const float old_delta = old_visits > 0 ? eval - old_eval : 0.0f;
    const float new_delta = eval - (old_eval + eval) / (old_visits + 1);

    // Welford's online algorithm for calculating variance.
    const float delta = old_delta * new_delta;

    m_visits.fetch_add(1);
    Utils::atomic_add(m_squared_eval_diff, delta);
    Utils::atomic_add(m_accumulated_red_stmevals, evals.red_stmeval);
    Utils::atomic_add(m_accumulated_red_wls, evals.red_winloss);
    Utils::atomic_add(m_accumulated_draws, evals.draw);
}

void UCTNode::increment_threads() {
    m_loading_threads.fetch_add(1);
}

void UCTNode::decrement_threads() {
    m_loading_threads.fetch_sub(1);
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
    return m_parameters;
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
        std::this_thread::yield();
    }
}
