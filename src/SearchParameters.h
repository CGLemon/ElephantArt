#ifndef SEARCHPARAMETERS_H_INCLUDE
#define SEARCHPARAMETERS_H_INCLUDE

#include "config.h"

class SearchParameters {
public:
    SearchParameters();
    int playouts;
    int random_min_visits;

    bool dirichlet_noise;
    bool ponder;
    bool collect;

    double fpu_root_reduction;
    double fpu_reduction;
    double logconst;
    double logpuct;
    double cpuct;
};

#endif
