#ifndef INSTANCE_H_INCLUDE
#define INSTANCE_H_INCLUDE
#include "Position.h"

class Instance {
public:
    enum Result {
        NONE = 0, DRAW, UNKNOWN
    };

    Instance(Position &position);

    Result judge();

private:
    Position m_position;

};

#endif
