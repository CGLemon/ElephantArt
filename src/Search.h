#ifndef SEARCH_H_INCLUDE
#define SEARCH_H_INCLUDE

#include <memory>
#include <functional>

#include "SearchParameters.h"
#include "ThreadPool.h"
#include "Search.h"
#include "Network.h"
#include "Position.h"
#include "UCTNode.h"
#include "Train.h"
#include "Utils.h"
#include "config.h"

class SearchResult {
public:
    SearchResult() = default;
    bool valid() const { return m_nn_evals != nullptr; }
    std::shared_ptr<UCTNodeEvals> nn_evals() const { return m_nn_evals; }

    void from_nn_evals(UCTNodeEvals nn_evals) { 
        m_nn_evals = std::make_shared<UCTNodeEvals>(nn_evals);;
    }

    void from_gameover(Position &position) {
        if (m_nn_evals == nullptr) return;
        m_nn_evals = std::make_shared<UCTNodeEvals>();

        const auto winner = position.get_winner();
        assert(winner != Types::INVALID_COLOR);
        if (winner == Types::RED) {
            m_nn_evals->red_stmeval = 1.0f;
            m_nn_evals->red_winloss = 1.0f;
            m_nn_evals->draw = 0.0f;
        } else if (winner == Types::BLACK) {
            m_nn_evals->red_stmeval = -1.0f;
            m_nn_evals->red_winloss = -1.0f;
            m_nn_evals->draw = 0.0f;
        } else if (winner == Types::EMPTY_COLOR) {
            m_nn_evals->red_stmeval = 0.0f;
            m_nn_evals->red_winloss = 0.0f;
            m_nn_evals->draw = 1.0f;
        }
    }

private:
    std::shared_ptr<UCTNodeEvals> m_nn_evals{nullptr};

};

struct SearchInfo {
    Move move;
};

class Search {
public:
    static constexpr auto MAX_PLAYOUTS = 150000;
    Search(Position &position, Network &network, Train &train);
    ~Search();

    SearchInfo nn_direct();
    SearchInfo random_move();
    SearchInfo uct_search();

private:
    void prepare_uct(std::ostream &out);
    void clear_nodes();
    void increment_playouts();
    void play_simulation(Position &currposition, UCTNode *const node,
                         UCTNode *const root_node, SearchResult &search_result);
    float get_min_psa_ratio();
    bool is_uct_running();
    void set_running(bool is_running);
    void set_playouts(int playouts);
    bool stop_thinking() const;
    Move uct_best_move() const;
    Position m_rootposition;
    Position & m_position;
    Network & m_network;
    Train & m_train;
    UCTNode * m_rootnode{nullptr}; 

    ThreadPool m_searchpool;
    std::unique_ptr<ThreadGroup<void>> m_threadGroup{nullptr};
    std::shared_ptr<UCTNodeStats> m_nodestats{nullptr};

    int m_maxplayouts;
    int m_maxvisits;
    std::atomic<bool> m_running{false};
    std::atomic<int> m_playouts{0};
    Utils::Timer m_timer;
    std::shared_ptr<SearchParameters> m_parameters{nullptr};
};
#endif
