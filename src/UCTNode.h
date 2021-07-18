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

#ifndef UCTNODE_H_INCLUDE
#define UCTNODE_H_INCLUDE

#define VIRTUAL_LOSS_COUNT (3)

#include "SearchParameters.h"
#include "Network.h"
#include "Position.h"
#include "NodePointer.h"
#include "Board.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

class UCTNode;

struct UCTNodeStats {
    std::atomic<int> nodes{0};
    std::atomic<int> edges{0};
};

struct UCTNodeData {
    float policy{0.0f};
    int maps{-1};
    std::shared_ptr<SearchParameters> parameters{nullptr};
    std::shared_ptr<UCTNodeStats> node_status{nullptr};
    UCTNode *parent{nullptr};
};

struct UCTNodeEvals {
    float red_stmeval{0.0f};
    float red_winloss{0.0f};
    float draw{0.0f}; 
};

class UCTNode {
public:
    using UCTNodePointer = NodePointer<UCTNode, UCTNodeData>;
    UCTNode(std::shared_ptr<UCTNodeData> data);
    ~UCTNode();

    UCTNodeEvals prepare_root_node(Network &network,
                                   Position &position);
    bool expend_children(Network &network,
                         Position &position,
                         const float min_psa_ratio,
                         const bool is_root = false);

    UCTNodeEvals get_node_evals() const;

    void policy_target_pruning();
    int get_maps() const;
    float get_policy() const;
    int get_visits() const;
    Types::Color get_color() const;

    float get_eval_variance(const float default_var, const int visits) const;
    float get_eval_lcb(const Types::Color color) const;
    std::vector<std::pair<float, int>> get_lcb_list(const Types::Color color);
    std::vector<std::pair<float, int>> get_winrate_list(const Types::Color color);
    int get_best_move();
    int randomize_first_proportionally(float random_temp);

    const std::vector<std::shared_ptr<UCTNodePointer>> &get_children() const;
    float get_stmeval(const Types::Color color,
                      const bool use_virtual_loss) const;
    float get_winloss(const Types::Color color,
                      const bool use_virtual_loss) const;
    float get_draw() const;
    float get_meaneval(const Types::Color color,
                       const bool use_virtual_loss = true) const;
    UCTNode *get_child(const int maps);
    UCTNode *get();

    UCTNode *uct_select_child(const Types::Color color,
                              const bool is_root);

    UCTNode *prob_select_child();

    void apply_evals(std::shared_ptr<UCTNodeEvals> evals);
    void update(std::shared_ptr<UCTNodeEvals> evals);

    void increment_threads();
    void decrement_threads();

    void make_terminated(Types::Color color);

    void set_visits(int v);
    void set_active(const bool active);
    void invalinode();

    bool has_children() const;
    bool expandable() const;
    bool is_expending() const;
    bool is_expended() const;
    bool is_pruned() const;
    bool is_active() const;
    bool is_valid() const;

    bool is_terminated() const;

    std::shared_ptr<UCTNodeStats> node_status() const;

private:
    float m_red_stmeval{0.0f};
    float m_red_winloss{0.0f};
    float m_draw{0.0f};

    bool m_terminated{false};

    Types::Color m_color{Types::INVALID_COLOR};
    
    std::atomic<int> m_visits{0};
    std::atomic<int> m_loading_threads{0};

    std::atomic<float> m_squared_eval_diff{1e-4f};
    std::atomic<float> m_accumulated_red_stmevals{0.0f};
    std::atomic<float> m_accumulated_red_wls{0.0f};
    std::atomic<float> m_accumulated_draws{0.0f};

    std::shared_ptr<UCTNodeData> m_data{nullptr};

    std::vector<std::shared_ptr<UCTNodePointer>> m_children;
    std::shared_ptr<SearchParameters> parameters() const;
    
    void link_nodelist(std::vector<Network::PolicyMapsPair> &nodelist, float min_psa_ratio);
    void link_nn_output(const Network::Netresult &raw_netlist,
                        const Types::Color color);
    void inflate_all_children();
    void release_all_children();

    void apply_dirichlet_noise(const float alpha);
    void set_policy(const float p);

    float get_uct_policy(std::shared_ptr<UCTNodePointer> child, bool noise) const;
    int get_threads() const;
    int get_virtual_loss() const;

    float get_nn_stmeval(const Types::Color color) const;
    float get_nn_winloss(const Types::Color color) const;
    float get_nn_meaneval(const Types::Color color) const;
    float get_nn_draw() const;
    float get_accumulated_evals() const;
    float get_accumulated_wls() const;
    float get_accumulated_draws() const;

    void increment_nodes();
    void decrement_nodes();

    void increment_edges();
    void decrement_edges();

    void inflate(std::shared_ptr<UCTNodePointer> child);
    void release(std::shared_ptr<UCTNodePointer> child);

    void set_result(Types::Color color);

    enum Status : std::uint8_t {
        INVALID,  // INVALID means that the node is illegal.
        PRUNED,
        ACTIVE
    };
    std::atomic<Status> m_status{ACTIVE};

    enum class ExpandState : std::uint8_t {
        INITIAL = 0,
        EXPANDING,
        EXPANDED
    };
    std::atomic<ExpandState> m_expand_state{ExpandState::INITIAL};

    // INITIAL -> EXPANDING
    bool acquire_expanding();

    // EXPANDING -> DONE
    void expand_done();

    // EXPANDING -> INITIAL
    void expand_cancel();

    // wait until we are on EXPANDED state
    void wait_expanded() const ;
};

class UCT_Information {
public:
  static size_t get_memory_used(UCTNode *node);

  static std::string get_stats_string(UCTNode *node, Position &position);

  static std::string get_memory_string(UCTNode *node);

  static std::string get_pvsrting(UCTNode *node);

  static std::vector<int> get_pvlist(UCTNode *node);
};

#endif
