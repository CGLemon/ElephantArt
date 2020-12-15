#ifndef UCTNODE_H_INCLUDE
#define UCTNODE_H_INCLUDE

#define VIRTUAL_LOSS_COUNT (2)

#include "SearchParameters.h"
#include "Network.h"
#include "Position.h"
#include "NodePointer.h"
#include "Board.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#define VIRTUAL_LOSS_COUNT (2)

struct UCTNodeData {
    float policy{0.0f};
    int maps{-1};
    std::shared_ptr<SearchParameters> parameters{nullptr};
};


class UCTNode {
public:
    using UCTNodePointer = NodePointer<UCTNode, UCTNodeData>;
    UCTNode(std::shared_ptr<UCTNodeData> data);
    ~UCTNode();

    bool expend_children(Network &network,
                         Position &position,
                         const float min_psa_ratio,
                         const bool is_root = false);
    
    int get_maps() const;
    float get_policy() const;
    int get_visits() const;
    int get_color() const;
    float get_raw_evaluation(const int color) const;
    float get_accumulated_evals() const;
    
    UCTNode *get();
    
private:
    // data
    float m_raw_red_stmeval{0.f};
    float m_raw_red_eval{0.f};
    float m_raw_black_eval{0.f};
    float m_raw_draw_eval{0.f};
    Types::Color m_color{Types::INVALID_COLOR};
    
    std::atomic<int> m_visits{0};
    std::atomic<int> m_loading_threads{0};
    std::atomic<float> m_squared_eval_diff{1e-4f};
    std::atomic<float> m_accumulated_red_stmevals{0.0f};
    std::atomic<float> m_accumulated_red_evals{0.0f};
    std::atomic<float> m_accumulated_blackevals{0.0f};
    std::atomic<float> m_accumulated_draw_evals{0.0f};
    
    void link_nodelist(std::vector<Network::PolicyMapsPair> &nodelist, float min_psa_ratio);
    void link_nn_output(const Network::Netresult &raw_netlist,
                        const Types::Color color);
    void increment_threads();
    
    std::vector<std::shared_ptr<UCTNodePointer>> m_children;
    std::shared_ptr<UCTNodeData> m_data{nullptr};
    std::shared_ptr<SearchParameters> parameters() const;

    enum Status : std::uint8_t {
        INVALID,
        // INVALID means that the node is illegal.
        PRUNED,
        ACTIVE
    };
    std::atomic<Status> m_status{ACTIVE};
    
    
    enum class ExpandState : std::uint8_t {
        INITIAL = 0,
        EXPANDING,
        EXPANDED,
        UPDATE
    };
    std::atomic<ExpandState> m_expand_state{ExpandState::INITIAL};

    // INITIAL -> EXPANDING
    bool acquire_expanding();

    // EXPANDING -> DONE
    void expand_done();

    // EXPANDING -> INITIAL
    void expand_cancel();

    // wait until we are on EXPANDED state
    void wait_expanded();
};

#endif
