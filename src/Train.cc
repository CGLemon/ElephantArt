/*
    This file is part of ElephantArt.
    Copyright (C) 2021 Hung-Zhe Lin

    ElephantArt is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElephantArt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElephantArt.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "Train.h"
#include "config.h"
#include "Decoder.h"

#include <algorithm>

template<typename T>
void vector_stream(const std::vector<T> arr, std::ostream &out) {
    std::for_each(std::begin(arr), std::end(arr),
               [&, cnt = size_t{0}, size = arr.size()](auto element) mutable {
                   out << element;
                   if (++cnt == size) {
                       Utils::strip_stream(out, 1);
                   } else {
                       Utils::space_stream(out, 1);
                   }
               }
    );
}

void DataCollection::out_stream(std::ostream &out) {
/*
 * ------- claiming -------
 * L1        : Version
 *
 * ------- Inputs data -------
 * L2  - L15 : Pieces Index
 * L16       : Current Player
 * L17       : Game plies 
 * L18       : Repeat conut
 * L19 - L20 : last move
 *
 * ------- Prediction data -------
 * L21       : Probabilities
 * L22       : Which pieces go to move
 * L23       : Result
 *
 */
    out << version << std::endl;

    for (const auto &past: pieces_history) {
        vector_stream(past.pawns[Types::RED], out);
        vector_stream(past.cannons[Types::RED], out);
        vector_stream(past.rooks[Types::RED], out);
        vector_stream(past.horses[Types::RED], out);
        vector_stream(past.elephants[Types::RED], out);
        vector_stream(past.advisors[Types::RED], out);
        vector_stream(past.kings[Types::RED], out);

        vector_stream(past.pawns[Types::BLACK], out);
        vector_stream(past.cannons[Types::BLACK], out);
        vector_stream(past.rooks[Types::BLACK], out);
        vector_stream(past.horses[Types::BLACK], out);
        vector_stream(past.elephants[Types::BLACK], out);
        vector_stream(past.advisors[Types::BLACK], out);
        vector_stream(past.kings[Types::BLACK], out);
    }

    out << (to_move == Types::RED ? 1 : 0) << std::endl;
    out << gameply << std::endl;
    out << repeat << std::endl;
    out << last_from_move << std::endl;
    out << last_to_move << std::endl;

    const auto p_s = probabilities.size();
    for (auto idx = size_t{0}; idx < p_s; ++idx) {
        const auto x = probabilities[idx];
        const auto mapes = x.first;
        const auto policy = x.second;

        out << mapes << " " << policy;

        if (idx == p_s-1) {
            Utils::strip_stream(out, 1);
        } else {
            Utils::space_stream(out, 1);
        }
    }

    Board::piece_stream<Types::ASCII>(out, static_cast<Types::Piece>(piece));
    Utils::strip_stream(out, 1);

    if (winner == Types::INVALID_COLOR) {
        out << "NA";
    } else if (winner == Types::EMPTY_COLOR) {
        out << "0";
    } else if (winner == to_move) {
        out << "1";
    } else if (winner != to_move) {
        out << "-1";
    };
    Utils::strip_stream(out, 1);
}


Train::Train() {
    m_counter = 0;
}

Types::Piece_t maps2piece(int maps, Position &pos) {
    const auto move = Decoder::maps2move(maps);
    const auto from = move.get_from();
    const auto piece = pos.get_piece_type(from);
    assert(Decoder::maps_valid(maps));
    assert(piece != Types::EMPTY_PIECE_T);
    return piece;
}

void proccess_probabilities(UCTNode &node, DataCollection &data, int min_cutoff) {
    assert(data.probabilities.empty());

    const auto children = node.get_children();
    auto temp = std::vector<std::pair<int, int>>{};

    for (const auto &child : children) {
        const auto maps = child->get()->get_maps();
        const auto visits = child->get()->get_visits();
        if (visits > min_cutoff) {
            temp.emplace_back(maps, visits);
        }
    }

    if (temp.empty()) {
        assert(min_cutoff != 0);
        proccess_probabilities(node, data, 0);
        return;
    }

    const auto acc_visits = std::accumulate(std::begin(temp), std::end(temp), 0,
                                [](int init, std::pair<int, int> x) { return init + x.second; }
                            );

    for (const auto &x : temp) {
        const auto maps = x.first;
        const auto visits = x.second;
        const auto probability = static_cast<float>(visits) / static_cast<float>(acc_visits);
        data.probabilities.emplace_back(maps, probability);
    }
}

void proccess_inputs(Position &pos, DataCollection &data) {
    assert(data.pieces_history.empty());
    auto &history = data.pieces_history;
    for (auto p = 0; p < INPUT_MOVES; ++p) {
        history.emplace_back(DataCollection::PositionPieces{});
        for (auto idx = size_t{0}; idx < Board::INTERSECTIONS; ++idx) {
            const auto x = idx % Board::WIDTH;
            const auto y = idx / Board::WIDTH;
            const auto vtx = Board::get_vertex(x, y);
            const auto pis = pos.get_piece(vtx);

            if (pis == Types::R_PAWN) {
                history[p].pawns[Types::RED].emplace_back(idx);
            } else if (pis == Types::R_CANNON) {
                history[p].cannons[Types::RED].emplace_back(idx);
            } else if (pis == Types::R_ROOK) {
                history[p].rooks[Types::RED].emplace_back(idx);
            } else if (pis == Types::R_HORSE) {
                history[p].horses[Types::RED].emplace_back(idx);
            } else if (pis == Types::R_ELEPHANT) {
                history[p].elephants[Types::RED].emplace_back(idx);
            } else if (pis == Types::R_ADVISOR) {
                history[p].advisors[Types::RED].emplace_back(idx);
            } else if (pis == Types::R_KING) {
                history[p].kings[Types::RED].emplace_back(idx);
            } else if (pis == Types::B_PAWN) {
                history[p].pawns[Types::BLACK].emplace_back(idx);
            } else if (pis == Types::B_CANNON) {
                history[p].cannons[Types::BLACK].emplace_back(idx);
            } else if (pis == Types::B_ROOK) {
                history[p].rooks[Types::BLACK].emplace_back(idx);
            } else if (pis == Types::B_HORSE) {
                history[p].horses[Types::BLACK].emplace_back(idx);
            } else if (pis == Types::B_ELEPHANT) {
                history[p].elephants[Types::BLACK].emplace_back(idx);
            } else if (pis == Types::B_ADVISOR) {
                history[p].advisors[Types::BLACK].emplace_back(idx);
            } else if (pis == Types::B_KING) {
                history[p].kings[Types::BLACK].emplace_back(idx);
            }
            assert(pis != Types::INVAL_PIECE);
        }
    }

    const auto to_move = pos.get_to_move();
    data.to_move = to_move;

    const auto inputs = Model::gather_planes(&pos, Board::IDENTITY_SYMMETRY);
    assert(inputs.size() == data.input_planes.size());
    std::copy(std::begin(inputs), std::end(inputs), std::begin(data.input_planes));

    const auto features = Model::gather_features(&pos);
    assert(features.size() == data.input_features.size());
    std::copy(std::begin(features), std::end(features), std::begin(data.input_features));

    data.movenum = pos.get_movenum();
    data.gameply = pos.get_gameply();
    data.repeat = pos.get_repeat().first;

    const auto lastmove = pos.get_last_move();
    if (lastmove.valid()) {
        const auto from = Board::get_xy(lastmove.get_from());
        const auto to = Board::get_xy(lastmove.get_to());
        data.last_from_move = Board::get_index(from.first, from.second);
        data.last_to_move = Board::get_index(to.first, to.second);
    } else {
        data.last_from_move = -1;
        data.last_to_move = -1;
    }
}

void Train::gather_probabilities(UCTNode &node, Position &pos) {
    if (!option<bool>("collect")) return;

    auto data = DataCollection{};
    data.version = get_version();
    proccess_probabilities(node, data, option<int>("min_cutoff"));

    const auto maps = node.get_best_move();
    const auto piece = maps2piece(maps, pos);
    data.piece = piece;

    proccess_inputs(pos, data);

    push_buffer(data);
}

void Train::gather_move(Move move, Position &pos) {
    if (!option<bool>("collect")) return;

    auto data = DataCollection{};
    data.version = get_version();
    const auto maps = Decoder::move2maps(move);
    data.probabilities.emplace_back(maps, 1.0f);

    const auto piece = maps2piece(maps, pos);
    data.piece = piece;

    proccess_inputs(pos, data);

    push_buffer(data);
}

void Train::gather_winner(Types::Color color) {
    if (!option<bool>("collect")) return;

    for (const auto &data : m_buffer) {
        assert(data->winner == Types::INVALID_COLOR);
        data->winner = color;
    }
}


void Train::data_stream(std::ostream &out) {
    for (const auto &data : m_buffer) {
        data->out_stream(out);
    }
}

void Train::save_data(std::string filename, bool append) {
    auto out = std::ostringstream{};
    data_stream(out);

    auto ios_tag = std::ios::out;

    if (append) {
        ios_tag |= std::ios::app;
    }

    std::fstream save_file;
  
    save_file.open(filename, ios_tag);
    if (save_file.is_open()) {
        save_file << out.str();
        save_file.close();
        clear_buffer();
    }
}

int Train::get_version() const {
    return 1;
}

void Train::push_buffer(DataCollection &data) {
    m_buffer.emplace_back(std::make_shared<DataCollection>(data));
    m_counter++;

    while (m_counter > option<int>("collection_buffer_size")) {
        m_buffer.pop_front();
        m_counter--;
    }
}

void Train::clear_buffer() {
    while (!m_buffer.empty()) {
        m_buffer.pop_front();
        m_counter--;
    }
    assert(m_counter == 0);
}
