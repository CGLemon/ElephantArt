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

#ifndef NODEPOINTER_H_INCLUDE
#define NODEPOINTER_H_INCLUDE

#include <memory>
#include <atomic>
#include <vector>
#include <cstdint>
#include <thread>

#define POINTER_MASK (3ULL)

static constexpr std::uint64_t UNINFLATED = 2ULL;
static constexpr std::uint64_t INFLATING = 1ULL;
static constexpr std::uint64_t POINTER = 0ULL;

template<typename Node, typename Data>
class NodePointer {
public:
    NodePointer() = default;
    NodePointer(std::shared_ptr<Data> data);
    NodePointer(const NodePointer &) = delete;
    NodePointer& operator=(const NodePointer&);

    ~NodePointer();

    bool is_pointer() const;
    bool is_inflating() const;
    bool is_uninflated() const;

    Node *read_ptr(uint64_t v) const;
    Node *get() const;

    bool inflate();
    bool release();

    std::shared_ptr<Data> data() const;

private:
    bool acquire_inflating();

    std::shared_ptr<Data> m_data{nullptr};
    std::atomic<std::uint64_t> m_pointer{UNINFLATED};

    bool is_pointer(std::uint64_t v) const;
    bool is_inflating(std::uint64_t v) const;
    bool is_uninflated(std::uint64_t v) const;
};

template<typename Node, typename Data>
inline NodePointer<Node, Data>::NodePointer(std::shared_ptr<Data> data) {
    m_data = data;
}

template<typename Node, typename Data>
inline NodePointer<Node, Data>::~NodePointer() {
    release();
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::is_pointer(std::uint64_t v) const {
    return (v & POINTER_MASK) == POINTER;
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::is_inflating(std::uint64_t v) const {
    return (v & POINTER_MASK) == INFLATING;
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::is_uninflated(std::uint64_t v) const {
    return (v & POINTER_MASK) == UNINFLATED;
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::is_pointer() const {
    return is_pointer(m_pointer.load());
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::is_inflating() const {
    return is_inflating(m_pointer.load());
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::is_uninflated() const {
    return is_uninflated(m_pointer.load());
}

template<typename Node, typename Data>
inline Node *NodePointer<Node, Data>::read_ptr(uint64_t v) const {
    assert(is_pointer(v));
    return reinterpret_cast<Node *>(v & ~(POINTER_MASK));
}

template<typename Node, typename Data>
inline Node *NodePointer<Node, Data>::get() const {
    auto v = m_pointer.load();
    if (is_pointer(v))
        return read_ptr(v);
    return nullptr;
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::acquire_inflating() {
    auto uninflated = UNINFLATED;
    auto newval = INFLATING;
    return m_pointer.compare_exchange_strong(uninflated, newval);
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::inflate() {
    while (true) {
        auto v = m_pointer.load();
        if (is_pointer(v)) {
            return false;
        }
        if (!acquire_inflating()) {
            std::this_thread::yield();
            continue;
        }
        auto new_pointer =
            reinterpret_cast<std::uint64_t>(new Node(m_data)) |
            POINTER;
        auto old_pointer = m_pointer.exchange(new_pointer);
#ifdef NDEBUG
        (void) old_pointer;
#endif
        assert(is_inflating(old_pointer));
        return true;
    }
}

template<typename Node, typename Data>
inline bool NodePointer<Node, Data>::release() {
    auto v = m_pointer.load();
    if (is_pointer(v)) {
        delete read_ptr(v);
        auto pointer = m_pointer.exchange(UNINFLATED);
#ifdef NDEBUG
        (void) pointer;
#endif
        assert(pointer == v);
        return true;
    }
    return false;
}

template<typename Node, typename Data>
inline std::shared_ptr<Data> NodePointer<Node, Data>::data() const {
    return m_data;
}

#endif
