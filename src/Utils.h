/*
    This file is part of Saya.
    Copyright (C) 2020 Hung-Zhe Lin

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

#include <cstdarg>
#include <thread>
#include <iostream>
#include <cassert>
#include <sstream>
#include <atomic>
#include <utility>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <initializer_list>

namespace Utils {

enum Printf_t {
    SYNC, STATIC, AUTO, EXTERN
};

/**
 * SYNC   : Printing the verbose to the terminate. It also save verbose
 *          in the log file, if the log file exist.
 *
 * STATIC : If the log file exixt, Saving verbose in the log file. If NOT,
 *          Printing the verbose to the terminate.
 *
 * AUTO   : It based on STATIC mode. The difference is the parameter
 *          "quit_verbose". If the "quit_verbose" is true, it will don't
 *          output any verbose.
 *
 * EXTERN : Only Saving verbose in the log file, if the log file exist.
 *
 */


template <Printf_t>
void printf_base(const char *fmt, va_list va);

template <Printf_t T>
void printf(const char *fmt, ...) {
    va_list va;
    va_start(va, fmt);
    printf_base<T>(fmt, va);
    va_end(va);
}

template <Printf_t>
void printf(std::ostringstream &out);

void space_stream(std::ostream &out, const size_t times);
void strip_stream(std::ostream &out, const size_t times);

float cached_t_quantile(int v);

template <typename T> 
void adjust_range(T &a, const T max, const T min = (T)0) {
    assert(max > min);
    if (a > max) {
        a = max;
    }
    else if (a < min) {
        a = min;
    }
}

template <typename T> 
void atomic_add(std::atomic<T> &f, T d) {
    T old = f.load();
    while (!f.compare_exchange_weak(old, old + d)) {
        std::this_thread::yield();
    }
}

/**
 * Transform the string to words.
 */
class CommandParser {
public:
    struct Reuslt {
        Reuslt(const std::string &s, const int i) : str(s), idx(i) {};

        Reuslt(const std::string &&s, const int i) :
            str(std::forward<decltype(s)>(s)), idx(i) {};

        std::string str;
        int idx;

        template<typename T> T get() const;
    };

    static constexpr size_t MAX_BUFFER_SIZE = 1024 * 1024 * 1024;

    CommandParser() = delete;
    CommandParser(std::string &input);
    CommandParser(std::string &input, const size_t max);
    CommandParser(int argc, char** argv);

    bool valid() const;
    size_t get_count() const;

    std::shared_ptr<Reuslt> get_command(size_t id) const;
    std::shared_ptr<Reuslt> get_commands(size_t begin = 0) const;
    std::shared_ptr<Reuslt> get_slice(size_t begin, size_t end) const;
    std::shared_ptr<Reuslt> find(const std::string input, int id = -1) const;
    std::shared_ptr<Reuslt> find(const std::initializer_list<std::string> inputs, int id = -1) const;
    std::shared_ptr<Reuslt> find_next(const std::string input) const;
    std::shared_ptr<Reuslt> find_next(const std::initializer_list<std::string> inputs) const;
    std::shared_ptr<Reuslt> remove_command(size_t id);
    std::shared_ptr<Reuslt> remove_slice(size_t begin, size_t end);

private:
    std::vector<std::shared_ptr<const std::string>> m_commands;
    size_t m_count;

    void parser(std::string &input, const size_t max);
    void parser(std::string &&input, const size_t max);
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
        Char
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

    operator char() const {
        assert(m_type == type::Char);
        return char(m_value[0]);
    }

    operator std::string() const {
        assert(m_type == type::String);
        return m_value;
    }

    template<typename T>
    void adjust();

    bool boundary_valid() const;
    void option_handle() const;

public:
    Option() = default;

    void operator<<(const Option &&o) { *this = std::forward<decltype(o)>(o); }

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

class Timer {
public:
    Timer();
    void clock();  
   
    int get_duration_seconds() const;
    int get_duration_milliseconds() const;
    int get_duration_microseconds () const;
    float get_duration() const;

    void record();
    void release();
    float get_record_time(size_t) const;
    int get_record_count() const;
    const std::vector<float>& get_record() const;

private:
    std::chrono::steady_clock::time_point m_clock_time;
    std::vector<float> m_record;
    size_t record_count;
};

class BitIterator {
public :
    BitIterator() = delete;
    BitIterator(const size_t s);

    std::vector<bool> get() const;
    void set(std::uint64_t cnt);

    void next();
    void back();

private :
    bool bit_signed(size_t s) const;
    std::uint64_t m_cnt{0ULL};
    size_t m_size{0};
};

} // namespace Utils

#endif
