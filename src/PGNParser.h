#include "Position.h"

#include <fstream>
#include <iostream>
#include <string>

class PGNParser {
public:
    void save_pgn(std::string filename, Position &pos);
    void pgn_stream(std::ostream &out, Position &pos);

private:
    std::string from_position(Position &pos) const;
    
};
