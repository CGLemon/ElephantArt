#ifndef CMLInterface_H_INCLUDE
#define CMLInterface_H_INCLUDE

#include "Utils.h"

class CMLInterface {
protected:
    virtual void init() = 0;

    virtual void loop() = 0;

    virtual std::string execute(Utils::CommandParser &parser) = 0;
};

#endif
