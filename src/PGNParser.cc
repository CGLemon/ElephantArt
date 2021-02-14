#include "PGNParser.h"
#include "Utils.h"

#include <vector>
#include <sstream>
#include <cassert>

void PGNParser::save_pgn(std::string filename, Position &pos) {
    auto pgn = from_position(pos);
    auto file = std::ofstream{};
    file.open(filename, std::ios::out | std::ios::app);
    if (file.is_open()) {
        file << pgn;
        file.close();
    } else {
        Utils::printf<Utils::STATIC>("Couldn't Open file %s\n", filename.c_str());
    }
}

void PGNParser::pgn_stream(std::ostream &out, Position &pos) {
    out << from_position(pos);
}

std::string PGNParser::from_position(Position &pos) const {
    auto pgn = std::ostringstream{};

    pgn << "[Game \"Chinese Chess\"]" << std::endl;
    pgn << "[Event \"\"]" << std::endl;
    pgn << "[Site \"\"]" << std::endl;
    pgn << "[Date \"\"]" << std::endl;
    pgn << "[Round \"\"]" << std::endl;
    pgn << "[Site \"\"]" << std::endl;
    pgn << "[Red \"\"]" << std::endl;
    pgn << "[Black \"\"]" << std::endl;
    pgn << "[Result \"\"]" << std::endl;
    pgn << "[Opening \"\"]" << std::endl;
    pgn << "[ECCO \"\"]" << std::endl;

    auto &history = pos.get_history();
    assert(!history.empty());

    pgn << "[FEN \"";
    history[0]->fen_stream(pgn);
    pgn << "\"]" << std::endl;
    pgn << "[Format \"WXF\"]" << std::endl;
    
    if (history.size() >= 2) {
        auto move_cut = history[1]->get_movenum()-1;
        for (auto idx = size_t{0}; idx < history.size() - 1; ++idx) {
            const auto movenum = history[idx]->get_movenum();
            const auto wxf = history[idx+1]->get_wxfmove();
            if (move_cut+1 == movenum) {
                if (move_cut != history[1]->get_movenum()-1) {
                    pgn << std::endl;
                }
                move_cut++;
                pgn << movenum << ".";
            }
            pgn << wxf << " ";
        }
    }
    return pgn.str();
}
