/*
    This file is part of Saya.
    Copyright (C) 2020 Hung-Zhe, Lin

    Saya is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Saya is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Saya.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include "Utils.h"
#include "config.h"

namespace Utils {

void auto_printf(const char *fmt, ...) {

    if (cfg_quiet) {
        return;
    }

    va_list ap;
    va_start(ap, fmt);
    vfprintf(stdout, fmt, ap);
    va_end(ap);
}

void auto_printf(std::ostringstream &out) {

    if (cfg_quiet) {
        return;
    }

    std::cout << out.str();
}

void space_stream(std::ostream &out, const size_t times) {
    for (auto t = size_t{0}; t < times; ++t) {
        out << " ";
    }
}

void strip_stream(std::ostream &out, const size_t times) {
    for (auto t = size_t{0}; t < times; ++t) {
        out << std::endl;
    }
}



} // namespace Utils
