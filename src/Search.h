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
// #include "Trainer.h"
#include "Utils.h"
#include "config.h"

class SearchResult {
public:
    SearchResult() = default;
    bool valid() const { return m_nn_evals != nullptr; }
    std::shared_ptr<UCTNodeEvals> nn_evals() const { return m_nn_evals; }

    void from_nn_evals(std::shared_ptr<UCTNodeEvals> nn_evals) { 
        m_nn_evals = nn_evals;
    }

    void from_score(Position &position) {
        if (m_nn_evals == nullptr) return;
        m_nn_evals = std::make_shared<UCTNodeEvals>();
    }

private:
    std::shared_ptr<UCTNodeEvals> m_nn_evals{nullptr};

};

struct SearchInfo {
    Move move;
    std::string analysis;
};

class Search {
public:
    static constexpr auto MAX_PLAYOUTS = 150000;
    Search(Position &position, Network &network);
    ~Search();

    SearchInfo nn_direct();
    SearchInfo random_move();

private:
    Position m_rootposition;
    Position & m_position;
    Network & m_network;

    std::unique_ptr<ThreadGroup<void>> m_threadGroup{nullptr};

    int m_maxplayouts;
    std::atomic<bool> m_running;
    std::atomic<int> m_playouts;
    Utils::Timer m_timer;
    std::shared_ptr<SearchParameters> m_parameters{nullptr};
};
#endif
