#include "Instance.h"

Instance::Instance(Position &position) : m_position(position) {}


Instance::Result Instance::judge() {

    if (m_position.get_repeat() < 2) {
        return NONE;
    }

    return UNKNOWN;

}
