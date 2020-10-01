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

#include <vector>
#include <string>
#include <memory>

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

/**
 * Transform the string to words, and store one by one.
 */
class CommandParser {
public:
    CommandParser() = delete;

    CommandParser(std::string input);

    bool valid() const;

    size_t get_count() const;

    std::string get_command(size_t id) const;

    std::string get_commands() const;

    bool find(const std::string input, int id = -1) const;

    bool find(const std::vector<std::string> inputs, int id = -1) const;

    std::string find_next(const std::string input) const;

    std::string find_next(const std::vector<std::string> inputs) const;

private:
    std::vector<std::shared_ptr<const std::string>> m_commands;

    size_t m_count;

    void parser(std::string &input);
};


/**
 * Option stores parameters, maximal and minimal.
 * When we put a value in it. It will adjust the value automatically.
 */

class Option {
private:
    enum class type {
        Invalid,
        String,
        Bool,
        Integer,
        Float,
    };

    type m_type{type::Invalid};
    std::string m_value{};
    int m_max{0};
    int m_min{0};

    Option(type t, std::string val, int max, int min) :
               m_type(t), m_value(val), m_max(max), m_min(min) {}

    operator int() const {
        assert(m_type == type::Integer);
        return std::stoi(m_value);
    }

    operator bool() const {
        assert(m_type == type::Bool);
        return (m_value == "true");
    }

    operator float() const {
        assert(m_type == type::Float);
        return std::stof(m_value);
    }

    operator std::string() const {
        assert(m_type == type::String);
        return m_value;
    }

    bool boundary_valid() const;

    template<typename T>
    void adjust();

    void option_handle() const;

public:
    Option() = default;

    void operator<<(const Option& o) { *this = o; }

    // Get Option object.
    template<typename T>
    static Option setoption(T val, int max = 0, int min = 0);

    // Get the value. We need to assign type.
    template<typename T>
    T get() const;

    // Set the value.
    template<typename T>
    void set(T value);
};

// Adjust the value. Be sure the value is not bigger 
// than maximal and smaller than minimal.
template<typename T>
void Option::adjust() {
    if (!boundary_valid()) {
        return;
    }

    const auto upper = static_cast<T>(m_max);
    const auto lower = static_cast<T>(m_min);
    const auto val = (T)*this;

    if (val > upper) {
        set<T>(upper);
    } else if (val < lower) {
        set<T>(lower);
    }
}


} // namespace Utils

#endif
