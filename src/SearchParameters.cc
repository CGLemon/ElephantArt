#include "SearchParameters.h"

SearchParameters::SearchParameters() {
    reset();
}

void SearchParameters::reset() {
    playouts           = option<int>("playouts");
    random_min_visits  = option<int>("random_min_visits");

    dirichlet_noise    = option<bool>("dirichlet_noise");
    ponder             = option<bool>("ponder");
    collect            = option<bool>("collect");

    fpu_root_reduction = option<float>("fpu_root_reduction");
    fpu_reduction      = option<float>("fpu_reduction");

    cpuct_init         = option<float>("cpuct_init");
    cpuct_root_init    = option<float>("cpuct_root_init");
    cpuct_base         = option<float>("cpuct_base");
    cpuct_root_base    = option<float>("cpuct_root_base");

    draw_factor        = option<float>("draw_factor");
    draw_root_factor   = option<float>("draw_root_factor");
}
