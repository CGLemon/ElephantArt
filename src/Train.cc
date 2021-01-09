#include "Train.h"
#include "config.h"
#include "Decoder.h"

void DataCollection::out_stream(std::ostream &out) {
    const auto in_s = input_features.size();
    for (auto idx = size_t{0}; idx < in_s; ++idx) {
        const auto x = input_features[idx];
        if (x) {
            out << "1";
        } else {
            out << "0";
        }
        if (idx == in_s-1) {
            Utils::strip_stream(out, 1);
        } else {
            Utils::space_stream(out, 1);
        }
    }

    const auto p_s = probabilities.size();
    for (auto idx = size_t{0}; idx < p_s; ++idx) {
        const auto x = probabilities[idx];
        const auto mapes = x.first;
        const auto policy = x.second;

        out << mapes << " " << policy;

        if (idx == in_s-1) {
            Utils::strip_stream(out, 1);
        } else {
            Utils::space_stream(out, 1);
        }
    }

    Board::piece_stream<Types::ASCII>(out, static_cast<Types::Piece>(piece));

    if (winner == Types::INVALID_COLOR) {
        // do nothing...
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

Types::Piece_t maps2piece(int maps, Position &position) {
    const auto move = Decoder::maps2move(maps);
    const auto from = move.get_from();
    const auto piece = position.get_piece_type(from);
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

    const auto acc_visits = std::accumulate(std::begin(temp), std::begin(temp), 0,
                                [](int init, std::pair<int, int> x) { return init + x.second; }
                            );

    for (const auto &x : temp) {
        const auto maps = x.first;
        const auto visits = x.second;
        const auto probability = static_cast<float>(visits) / static_cast<float>(acc_visits);
        data.probabilities.emplace_back(maps, probability);
    }
}

void Train::gather_probabilities(UCTNode &node, Position &position) {
    if (!option<bool>("collect")) return;

    auto data = DataCollection{};
    proccess_probabilities(node, data, option<int>("min_cutoff"));

    const auto to_move = node.get_color();
    data.to_move = to_move;
    assert(to_move == position.get_to_move());

    const auto maps = node.get_maps();
    const auto piece = maps2piece(maps, position);
    data.piece = piece;

    const auto inputs = Model::gather_planes(&position, Board::IDENTITY_SYMMETRY);
    assert(inputs.size() == data.input_features.size());

    std::copy(std::begin(inputs), std::end(inputs), std::begin(data.input_features));
    push_buffer(data);
}

void Train::gather_move(Move move, Position &position) {

    if (!option<bool>("collect")) return;
    auto data = DataCollection{};
    const auto maps = Decoder::move2maps(move);
    data.probabilities.emplace_back(maps, 1.0f);

    const auto to_move = position.get_to_move();
    data.to_move = to_move;

    const auto piece = maps2piece(maps, position);
    data.piece = piece;

    const auto inputs = Model::gather_planes(&position, Board::IDENTITY_SYMMETRY);
    assert(inputs.size() == data.input_features.size());

    std::copy(std::begin(inputs), std::end(inputs), std::begin(data.input_features));
    push_buffer(data);
}

void Train::gather_winner(Types::Color color) {
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

void Train::push_buffer(DataCollection &data) {
    m_buffer.emplace_back(std::make_shared<DataCollection>(data));
    m_counter++;

    while (m_counter > option<int>("collect_buffer_size")) {
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
