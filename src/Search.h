#ifndef SEARCH_H_INCLUDE
#define SEARCH_H_INCLUDE

#include <memory>
#include <mutex>
#include <functional>
#include <limits>

#include "SearchParameters.h"
#include "TimeControl.h"
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
        m_nn_evals = std::make_shared<UCTNodeEvals>(nn_evals);
    }

    void from_gameover(Position &position) {
        if (m_nn_evals == nullptr) {
            m_nn_evals = std::make_shared<UCTNodeEvals>();
        }
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

class SearchInformation {
public:
    Move move;
    int depth;
    float seconds;
};

class SearchSetting {
public:
    bool ponder{false};
    int nodes{std::numeric_limits<int>::max()};
    int depth{std::numeric_limits<int>::max()};
    int milliseconds{std::numeric_limits<int>::max()};
    int movestogo{0};
    int increment{0};
};

class Search {
public:
    static constexpr auto MAX_PLAYOUTS = 1500000;
    Search(Position &pos, Network &network, Train &train);
    ~Search();

    Move nn_direct_move();
    Move random_move();
    Move uct_move();
    void think(SearchSetting setting, SearchInformation *info);
    void interrupt();
    void ponderhit();
    std::shared_ptr<SearchParameters> parameters();
    
private:
    void prepare_uct();
    void clear_nodes();
    void increment_playouts();
    void play_simulation(Position &currpos, UCTNode *const node,
                         UCTNode *const root_node, SearchResult &search_result, int &depth);
    float get_min_psa_ratio();
    bool is_running();
    void set_running(bool is_running);
    void set_playouts(int playouts);
    bool stop_thinking(int elapsed, int limittime) const;
    Move uct_best_move() const;

    void increment_threads();
    void decrement_threads();

    std::mutex m_thinking_mtx;
    SearchSetting m_setting;
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
    std::atomic<int> m_running_threads{0};
    std::atomic<bool> m_running{false};
    std::atomic<int> m_playouts{0};
    Utils::Timer m_timer;
    std::shared_ptr<SearchParameters> m_parameters{nullptr};
};
#endif
