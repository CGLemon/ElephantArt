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
    m_data = data;
    assert(m_data->parameters != nullptr);
}

UCTNode::~UCTNode() {
    assert(m_loading_threads.load() == 0);
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
    
    (void) is_root;
    
    auto movelist = position.get_movelist();
    for (const auto &move : movelist) {
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
            m_children.emplace_back(std::make_shared<UCTNodePointer>(data));
        }
    }
    assert(!m_children.empty());
}

void UCTNode::link_nn_output(const Network::Netresult &raw_netlist,
                             const Types::Color color){

    m_raw_draw_eval = raw_netlist.winrate[1];
    
    if (color == Types::RED) {
        m_raw_red_eval = raw_netlist.winrate[0];
        m_raw_black_eval = raw_netlist.winrate[2];
    } else {
        m_raw_red_eval = raw_netlist.winrate[2];
        m_raw_black_eval = raw_netlist.winrate[0];
    }
    
    m_raw_red_stmeval = m_raw_red_eval + m_raw_draw_eval * 0.5;
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

float UCTNode::get_raw_evaluation(const int color) const {
    if (color == Types::RED) {
        return m_raw_red_stmeval;
    }
    return 1.0f - m_raw_red_stmeval;
}

float UCTNode::get_accumulated_evals() const {
    return m_accumulated_red_stmevals.load();
}

int UCTNode::get_color() const {
   return m_color;
}

UCTNode *UCTNode::get() {
    return this;
}

void UCTNode::increment_threads() {
    m_loading_threads++;
}

std::shared_ptr<SearchParameters> UCTNode::parameters() const {
    return m_data->parameters;
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

void UCTNode::wait_expanded() {
    while (true) {
        auto v = m_expand_state.load();
        if (v == ExpandState::EXPANDED) {
            break;
        }
        std::this_thread::yield();
    }
}
