#include <numeric>

#include "Board.h"
#include "Search.h"
#include "UCTNode.h"
#include "Random.h"
#include "Model.h"
#include "Decoder.h"

ThreadPool SearchPool;

Search::Search(Position &position, Network &network) : 
    m_position(position),  m_network(network) {

    m_threadGroup = std::make_unique<ThreadGroup<void>>(SearchPool);
    m_parameters = std::make_shared<SearchParameters>();
}

Search::~Search() {

}

SearchInfo Search::nn_direct() {
    m_rootposition = m_position;
    auto eval = m_network.get_output(&m_rootposition, Network::Ensemble::NONE);
    int best_maps = -1;
    float best_policy = std::numeric_limits<float>::lowest();

    for (int m = 0; m < POLICYMAP * Board::INTERSECTIONS; ++m) {
        if (eval.policy[m] > best_policy && Decoder::maps_valid(m)) {
            if (!m_rootposition.is_legal(Decoder::maps2move(m))) {
                continue;
            }

            best_policy = eval.policy[m];
            best_maps = m;
        }
    }

    auto info = SearchInfo{};
    info.move = Decoder::maps2move(best_maps);

    return info;
}

SearchInfo Search::random_move() {
    m_rootposition = m_position;

    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    int maps = -1;

    while (true) {
        const auto randmaps = rng.randfix<POLICYMAP * Board::INTERSECTIONS>();

        if (Decoder::maps_valid(randmaps)) {
            continue;
        }

        if (m_rootposition.is_legal(Decoder::maps2move(randmaps))) {
            maps = randmaps;
            break;
        }
    }

    auto info = SearchInfo{};
    info.move = Decoder::maps2move(maps);

    return info;
}
