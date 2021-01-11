#ifndef TRAIN_H_INCLUDE
#define TRAIN_H_INCLUDE

#include "config.h"
#include "Board.h"
#include "Model.h"
#include "UCTNode.h"
#include "Types.h"
#include "Position.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <list>

struct DataCollection {
    using MAPS_PAIR = std::pair<int, float>;
    using PIECES_PAIR = std::array<std::vector<int>, 2>;


    struct PositionPieces {
        PIECES_PAIR pawns;
        PIECES_PAIR horses;
        PIECES_PAIR cannons;
        PIECES_PAIR rooks;
        PIECES_PAIR elephants;
        PIECES_PAIR advisors;
        PIECES_PAIR kings;
    };

    using PIECES_HISTORY = std::vector<PositionPieces>;
    int version;
    int movenum;
    int gameply;
    int repeat;
    std::array<float, Board::INTERSECTIONS * INPUT_CHANNELS> input_features;

    PIECES_HISTORY pieces_history;

    std::vector<MAPS_PAIR> probabilities;
    Types::Piece_t piece;

    Types::Color to_move{Types::INVALID_COLOR};
    Types::Color winner{Types::INVALID_COLOR};

    void out_stream(std::ostream &out);
};

class Train {
public:
    Train();

    void gather_probabilities(UCTNode &node, Position &position);
    void gather_move(Move move, Position &position);

    void gather_winner(Types::Color color);
    void save_data(std::string filename, bool append = true);
    void data_stream(std::ostream &out);

    void clear_buffer();

private:
    int get_version() const;
    void push_buffer(DataCollection &data);

    using Step = std::shared_ptr<DataCollection>;
    std::list<Step> m_buffer;
    int m_counter;

};

#endif
