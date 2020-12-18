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
    logconst           = option<float>("logconst");
    logpuct            = option<float>("logpuct");
    cpuct              = option<float>("cpuct");
}
