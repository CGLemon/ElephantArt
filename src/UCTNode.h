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
struct UCTNodeData {
    float policy{0.0f};
    int maps{-1};
    std::shared_ptr<SearchParameters> parameters{nullptr};
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
    int get_maps() const;
    float get_policy() const;
    int get_visits() const;
    int get_color() const;

    float get_eval_variance(const float default_var, const int visits) const;
    float get_eval_lcb(const Types::Color color) const;
    std::vector<std::pair<float, int>> get_lcb_list(const Types::Color color);
    std::vector<std::pair<float, int>> get_winrate_list(const Types::Color color);
    int get_best_move();
    int randomize_first_proportionally(float random_temp);

    UCTNode *get_child(const int maps);
    UCTNode *get();

    UCTNode *uct_select_child(const Types::Color color,
                              const bool is_root) const;
    
    void update(UCTNodeEvals &evals);

    void increment_threads();
    void decrement_threads();

    void set_active(const bool active);
    void invalinode();

    bool has_children() const;
    bool expandable() const;
    bool is_expending() const;
    bool is_expended() const;
    bool is_pruned() const;
    bool is_active() const;
    bool is_valid() const;

private:
    float m_red_stmeval{0.0f};
    float m_red_winloss{0.0f};
    float m_draw{0.0f};

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
    void dirichlet_noise(const float epsilon, const float alpha);
    void set_policy(const float p);

    int get_threads() const;
    int get_virtual_loss() const;

    float get_nn_stmeval(const Types::Color color) const;
    float get_nn_winloss(const Types::Color color) const;
    float get_nn_meaneval(const Types::Color color) const;
    float get_nn_draw() const;
    float get_accumulated_evals() const;
    float get_accumulated_wls() const;
    float get_accumulated_draws() const;

    float get_stmeval(const Types::Color color,
                      const bool use_virtual_loss) const;
    float get_winloss(const Types::Color color,
                      const bool use_virtual_loss) const;
    float get_meaneval(const Types::Color color,
                       const bool use_virtual_loss = true) const;
    float get_draw() const;

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

#endif
