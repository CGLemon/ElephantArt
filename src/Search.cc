#include <numeric>

#include "Board.h"
#include "Search.h"
#include "UCTNode.h"
#include "Random.h"

ThreadPool SearchPool;

Search::Search(Position &position, Network &network) : 
    m_position(position),  m_network(network) {

    m_threadGroup = std::make_unique<ThreadGroup<void>>(SearchPool);
    m_parameters = std::make_shared<SearchParameters>();
}

Search::~Search() {

}

