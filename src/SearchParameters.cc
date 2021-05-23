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

#include "SearchParameters.h"

SearchParameters::SearchParameters() {
    reset();
}

void SearchParameters::reset() {
    threads            = option<int>("threads");
    visits             = option<int>("visits");
    cap_playouts       = option<int>("cap_playouts");
    playouts           = option<int>("playouts");
    random_min_visits  = option<int>("random_min_visits");
    forced_checkmate_depth      = option<int>("forced_checkmate_depth");
    forced_checkmate_root_depth = option<int>("forced_checkmate_root_depth");

    dirichlet_noise    = option<bool>("dirichlet_noise");
    collect            = option<bool>("collect");

    draw_threshold     = option<float>("draw_threshold");
    resign_threshold   = option<float>("resign_threshold");
    fpu_root_reduction = option<float>("fpu_root_reduction");
    fpu_reduction      = option<float>("fpu_reduction");
    cpuct_init         = option<float>("cpuct_init");
    cpuct_root_init    = option<float>("cpuct_root_init");
    cpuct_base         = option<float>("cpuct_base");
    cpuct_root_base    = option<float>("cpuct_root_base");
    draw_factor        = option<float>("draw_factor");
    draw_root_factor   = option<float>("draw_root_factor");
    dirichlet_epsilon  = option<float>("dirichlet_epsilon");
    dirichlet_factor   = option<float>("dirichlet_factor");
    dirichlet_init     = option<float>("dirichlet_init");
}
