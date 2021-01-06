#include "Train.h"
#include "config.h"
#include "Decoder.h"

Train::Train() {
    m_counter = 0;
}

Types::Piece_t maps2piece(int maps, Position &position) {
    assert(Decoder::maps_valid(maps));
    const auto move = Decoder::maps2move(maps);
    const auto from = move.get_from();
    const auto piece = position.get_piece_type(from);
    return piece;
}

void proccess_probabilities(UCTNode &node, DataCollection &data) {

    assert(data.probabilities.empty());

    const auto total_visits = node.get_visits();
    const auto children = node.get_children();
    for (const auto &child : children) {
        const auto maps = child->get()->get_maps();
        const auto visits = child->get()->get_visits();
        const auto probability = static_cast<float>(visits) / static_cast<float>(total_visits);
        data.probabilities.emplace_back(maps, probability);
    }
}

void Train::gather_probabilities(UCTNode &node, Position &position) {
    if (!option<bool>("collect")) return;

    auto data = DataCollection{};
    proccess_probabilities(node, data);

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
        data->winner=color;
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
