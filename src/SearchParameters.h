#ifndef SEARCHPARAMETERS_H_INCLUDE
#define SEARCHPARAMETERS_H_INCLUDE

#include "config.h"

class SearchParameters {
public:
    SearchParameters();

    void reset();

    int threads;
    int visits;
    int playouts;
    int random_min_visits;

    bool using_traditional_chinese;
    bool dirichlet_noise;
    bool ponder;
    bool collect;

    float fpu_root_reduction;
    float fpu_reduction;
    float cpuct_init;
    float cpuct_root_init;
    float cpuct_base;
    float cpuct_root_base;
    float draw_factor;
    float draw_root_factor;
    float dirichlet_epsilon;
    float dirichlet_factor;
    float dirichlet_init;
};

#endif
