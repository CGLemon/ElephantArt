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

Search::Search(Position &position, Network &network, Train &train) : 
    m_position(position),  m_network(network), m_train(train) {

    m_parameters = std::make_shared<SearchParameters>();

    const auto t = m_parameters->threads-1;
    m_searchpool.initialize(t);
    m_threadGroup = std::make_unique<ThreadGroup<void>>(m_searchpool);

    m_maxplayouts = m_parameters->playouts;
    m_maxvisits = m_parameters->visits;
}

Search::~Search() {
    clear_nodes();
}

void Search::increment_playouts() {
    m_playouts.fetch_add(1);
}

float Search::get_min_psa_ratio() {
    auto v = m_playouts.load();
    if (v >= MAX_PLAYOUTS) {
        return 1.0f;
    }
    return 0.0f;
}

bool Search::is_uct_running() {
    return m_running.load();
}

void Search::set_running(bool is_running) {
    m_running.store(is_running);
}

void Search::set_playouts(int playouts) {
    m_playouts.store(playouts);
}

bool Search::stop_thinking() const {
   return m_playouts.load() > m_maxplayouts ||
              m_rootnode->get_visits() > m_maxvisits;
}

void Search::play_simulation(Position &currposition, UCTNode *const node,
                             UCTNode *const root_node, SearchResult &search_result) {

    node->increment_threads();

    if (node->expandable()) {
        if (currposition.gameover()) {
            search_result.from_gameover(currposition);
            node->apply_evals(search_result.nn_evals());
        } else {
            const bool has_children = node->has_children();
            const bool success = node->expend_children(m_network,
                                                       currposition,
                                                       get_min_psa_ratio());
            if (!has_children && success) {
                const auto nn_evals = node->get_node_evals();
                search_result.from_nn_evals(nn_evals);
            }
        }
    }

    if (node->has_children() && !search_result.valid()) {
        auto color = currposition.get_to_move();
        auto next = node->uct_select_child(color, node == root_node);
        auto maps = next->get_maps();
        auto move = Decoder::maps2move(maps);
        currposition.do_move_assume_legal(move);
        play_simulation(currposition, next, root_node, search_result);
    }

    if (search_result.valid()) {
        auto out = search_result.nn_evals();
        node->update(out);
    }

    node->decrement_threads();
}

SearchInfo Search::uct_search() {
    m_rootposition = m_position;
    const auto uct_worker = [&]() -> void {
        do {
            auto currposition = std::make_unique<Position>(m_rootposition);
            auto result = SearchResult{};
            play_simulation(*currposition, m_rootnode, m_rootnode, result);
            if (result.valid()) {
                increment_playouts();
            }
        } while(is_uct_running());
    };
    auto info = SearchInfo{};
    auto out = std::ostringstream{};

    auto timer = Utils::Timer{};

    prepare_uct();
    m_threadGroup->fill_tasks(uct_worker);

    bool keep_running = true;
    do {
        auto currposition = std::make_unique<Position>(m_rootposition);
        auto result = SearchResult{};

        play_simulation(*currposition, m_rootnode, m_rootnode, result);
        if (result.valid()) {
            increment_playouts();
        }

        keep_running &= (!stop_thinking());
        set_running(keep_running);
    } while (is_uct_running());

    m_threadGroup->wait_all();

    m_train.gather_probabilities(*m_rootnode, m_rootposition);

    info.move = uct_best_move();
    const auto s =timer.get_duration();
    Utils::printf<Utils::ANALYSIS>("Searching time %.4f second(s)\n", s);

    UCT_Information::dump_stats(m_rootnode, m_rootposition);
    clear_nodes();

    return info;
}

Move Search::uct_best_move() const {
    const auto maps = m_rootnode->get_best_move();
    return Decoder::maps2move(maps);
}

void Search::prepare_uct() {
    auto data = std::make_shared<UCTNodeData>();
    m_nodestats = std::make_shared<UCTNodeStats>();
    data->parameters = m_parameters;
    data->node_status = m_nodestats;
    m_rootnode = new UCTNode(data);

    set_playouts(0);
    set_running(true);
    m_rootnode->prepare_root_node(m_network, m_rootposition);

    const auto color = m_rootposition.get_to_move();
    const auto nn_eval = m_rootnode->get_node_evals();
    const auto stm_eval = color == Types::RED ? nn_eval.red_stmeval : 1 - nn_eval.red_stmeval;
    const auto winloss = color == Types::RED ? nn_eval.red_winloss : 1 - nn_eval.red_winloss;

    Utils::printf<Utils::ANALYSIS>("Raw NN output\n");
    Utils::printf<Utils::ANALYSIS>("  stm eval : %.2f%\n", stm_eval * 100.f);
    Utils::printf<Utils::ANALYSIS>("  winloss : %.2f%\n", winloss * 100.f);
    Utils::printf<Utils::ANALYSIS>("  draw probability : %.2f%\n", nn_eval.draw * 100.f);
}

void Search::clear_nodes() {
    if (m_rootnode) {
        delete m_rootnode;
        m_rootnode = nullptr;
    }
    if (m_nodestats) {
        assert(m_nodestats->nodes.load() == 0);
        assert(m_nodestats->edges.load() == 0);
        m_nodestats.reset();
        m_nodestats = nullptr;
    }
}

SearchInfo Search::nn_direct() {
    m_rootposition = m_position;
    auto analysis = std::vector<std::pair<float, int>>();
    auto acc = 0.0f;
    auto eval = m_network.get_output(&m_rootposition, Network::Ensemble::RANDOM_SYMMETRY);
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
    auto prec = option<int>("float_precision");
    for (int i = 0; i < 10; ++i) {
        const auto policy = analysis[i].first;
        const auto maps = analysis[i].second;
        const auto move = Decoder::maps2move(maps);
        out << "Move "
            << move.to_string()
            << " -> Policy : raw "
            << std::fixed
            << std::setprecision(prec)
            << policy
            << " | normalize "
            << std::fixed
            << std::setprecision(prec)
            << policy / acc
            << std::endl;
    }

    Utils::printf<Utils::ANALYSIS>(out);

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
