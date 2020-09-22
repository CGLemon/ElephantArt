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

#ifndef UTILS_H_DEFINED
#define UTILS_H_DEFINED

#include "config.h"

#include <iostream>
#include <cassert>
#include <sstream>
#include <atomic>


namespace Utils {

void auto_printf(const char *fmt, ...);

void auto_printf(std::ostringstream &out);

void space_stream(std::ostream &out, const size_t times);

void strip_stream(std::ostream &out, const size_t times);


template <class T> 
void atomic_add(std::atomic<T> &f, T d) {
  T old = f.load();
  while (!f.compare_exchange_weak(old, old + d)) {}
}

} // namespace Utils

#endif
