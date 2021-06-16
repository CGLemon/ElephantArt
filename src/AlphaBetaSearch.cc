#include "Search.h"
#include "Evaluate.h"

#include <algorithm>
#include <limits>

constexpr int MAX_SEARCH_DEPTH = 256;
constexpr int VALUE_MATE = 32000;

constexpr int mate_in(int ply) {
    return VALUE_MATE - ply;
}

constexpr int mated_in(int ply) {
    return -VALUE_MATE + ply;
}

int Search::get_root_depth(int count) {
    return 6 + count;
}

void Search::prepare_root_moves() {
    auto movelist = m_rootposition.get_movelist();
    m_generation += 1;
    m_root_moves.clear();
    auto color = m_rootposition.get_to_move();

    for (auto m : movelist) {
        m_rootposition.do_move_assume_legal(m);

        auto hash = m_rootposition.get_hash();
        bool tt_hit;
        auto tte = m_tt.probe(hash, tt_hit);
        auto root_move = RootMove(m);

        if (tt_hit) {
            root_move.score = tte->value;
        } else {
            auto value = Evaluate::get().calc_value(m_rootposition, color);
            tte->save(hash, Move{}, value, value, 0, m_generation);
            root_move.score = value;
            root_move.hash = hash;
        }

        m_root_moves.emplace_back(root_move);
        m_rootposition.undo_move();
    }

    sort_root_moves();
}

void Search::sort_root_moves() {
    std::stable_sort(std::begin(m_root_moves), std::end(m_root_moves),
                         [](auto &curr, auto &next) {
                             return curr.score > next.score;
                         }
                    );
}

void Search::alpha_beta_think(SearchInformation *info) {
    m_search_group->wait_all();
    m_rootposition = m_position;
    if (m_rootposition.gameover(true)) {
        return;
    }

    prepare_root_moves();

    const auto main_worker = [this, info]() -> void {
        auto stacks = std::vector<Stack>(MAX_SEARCH_DEPTH);
        for (int i = 0; i < MAX_SEARCH_DEPTH; ++i) {
            stacks[i].ply = i;
        }

        bool keep_running = true;
        int count = 0;

        const auto limitnodes = m_setting.nodes;
        const auto limitdepth = m_setting.depth;

        while(is_running()) {
            auto depth = get_root_depth(count++);
            auto currpos = std::make_unique<Position>(m_rootposition);
            auto value = stack_search(*currpos, true, stacks.data(),
                                          std::numeric_limits<int>::min(),
                                          std::numeric_limits<int>::max(), depth);


            for (auto &rm: m_root_moves) {
                bool tt_hit;
                auto tte = m_tt.probe(rm.hash, tt_hit);

                // Get the value from transposition table.
                if (tt_hit) {
                    rm.score = tte->value;
                }
            }

            sort_root_moves();

            keep_running &= (!(limitdepth < depth));
            keep_running &= is_running();
            set_running(keep_running);
            break;
        }


        if (info) {
            info->best_move = m_root_moves[0].move;
        }
    };

    set_running(true);
    m_search_group->add_task(main_worker);
}

int Search::stack_search(Position &currpos, bool pv, Stack *ss, int alpha, int beta, int depth) {
    auto hash = currpos.get_hash();
    auto tt_hit = bool{false}; 
    auto tte = m_tt.probe(hash, tt_hit);

    auto color = m_rootposition.get_to_move();

    if (tt_hit) {
        if (tte->depth > depth) {
            return tte->value;
        }
    }

    if (depth == 0) {
        if (tt_hit) {
            return tte->value;
        }

        auto value = Evaluate::get().calc_value(currpos, color);
        tte->save(hash, Move{}, value, value, 0, m_generation);
        return value;
    }


    int max_value = std::numeric_limits<int>::max();
    int min_value = std::numeric_limits<int>::min();
    int eval = 0;
    int best_value = min_value;
    Move best_move;

    auto movelist = currpos.get_movelist();

    for (auto m : movelist) {
        currpos.do_move_assume_legal(m);

        Stack * next_ss = ss+1;
        int value = -stack_search(currpos, pv, next_ss, alpha, beta, depth-1);

        if (best_value < value) {
            best_value = value;
            best_move = m;
        }

        currpos.undo_move();
    }

    if (tt_hit) {
        eval = tte->eval;
    }

    tte->save(hash, best_move, best_value, eval, depth, m_generation);

    return best_value;
}

int Search::stack_qsearch(Position &currpos, bool pv) {
    // Not yet.
    return 0;
}
