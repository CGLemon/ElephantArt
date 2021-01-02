#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "Board.h"
#include "Search.h"
#include "UCTNode.h"
#include "Random.h"
#include "Model.h"
#include "Decoder.h"
#include "config.h"

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
    auto analysis = std::vector<std::pair<float, int>>();
    auto acc = 0.0f;
    auto eval = m_network.get_output(&m_rootposition, Network::Ensemble::NONE);
    for (int m = 0; m < POLICYMAP * Board::INTERSECTIONS; ++m) {
        if (!Decoder::maps_valid(m)) {
            continue;
        }

        if (!m_rootposition.is_legal(Decoder::maps2move(m))) {
            continue;
        }
        analysis.emplace_back(eval.policy[m], m);
        acc += eval.policy[m];
    }

    std::stable_sort(std::rbegin(analysis), std::rend(analysis));

    auto info = SearchInfo{};
    info.move = Decoder::maps2move(std::begin(analysis)->second);

    auto out = std::ostringstream{};
    auto pres = option<int>("float_precision");
    for (int i = 0; i < 10; ++i) {
        const auto policy = analysis[i].first;
        const auto maps = analysis[i].second;
        const auto move = Decoder::maps2move(maps);
        out << "Move "
            << move.to_string()
            << " -> Policy : raw "
            << std::fixed
            << std::setprecision(pres)
            << policy
            << " | normalize "
            << std::fixed
            << std::setprecision(pres)
            << policy / acc
            << std::endl;
    }

    Utils::auto_printf(out);
    info.analysis = out.str();

    return info;
}

SearchInfo Search::random_move() {
    m_rootposition = m_position;

    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    int maps = -1;

    while (true) {
        const auto randmaps = rng.randfix<POLICYMAP * Board::INTERSECTIONS>();

        if (!Decoder::maps_valid(randmaps)) {
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
