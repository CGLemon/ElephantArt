#ifndef TRAIN_H_INCLUDE
#define TRAIN_H_INCLUDE

#include "config.h"
#include "Board.h"
#include "Model.h"
#include "UCTNode.h"
#include "Types.h"
#include "Position.h"

#include <memory>
#include <vector>
#include <list>

struct DataCollection {
     using MAPS_PAIR = std::pair<int, float>;
     std::array<bool, Board::INTERSECTIONS * INPUT_CHANNELS> input_features;

     std::vector<MAPS_PAIR> probabilities;
     Types::Piece_t piece;

     Types::Color to_move;
     Types::Color winner;
};

class Train {
public:
    Train();

    void gather_probabilities(UCTNode &node, Position &position);
    void gather_move(Move move, Position &position);

    void gather_winner(Types::Color color);

private:
    void push_buffer(DataCollection &data);
    void clear_buffer();

    using Data = std::shared_ptr<DataCollection>;
    std::list<Data> m_buffer;
    int m_counter;

};

#endif
