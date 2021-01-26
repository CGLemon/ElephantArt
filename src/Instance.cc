#include "Instance.h"

Instance::Instance(Position &position) : m_position(position) {}


Instance::Result Instance::judge() {

    auto repeat = m_position.get_repeat();
    if (repeat.first < 2) {
        return NONE;
    }

    return UNKNOWN;

}
