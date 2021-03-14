/*
    This file is part of ElephantArt.
    Copyright (C) 2021 Hung-Zhe Lin

    ElephantArt is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ElephantArt is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with ElephantArt.  If not, see <http://www.gnu.org/licenses/>.
*/

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

    bool dirichlet_noise;
    bool collect;

    float draw_threshold;
    float resign_threshold;
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
