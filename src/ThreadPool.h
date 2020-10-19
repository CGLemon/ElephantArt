/*
    Extended from code:
    Copyright (c) 2012 Jakob Progsch, Václav Zeman
    Copyright (c) 2017-2019 Gian-Carlo Pascutto and contributors
    Modifications:
    Copyright (c) 2020 Hung Zhe, Lin

    This software is provided 'as-is', without any express or implied
    warranty. In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
    claim that you wrote the original software. If you use this software
    in a product, an acknowledgment in the product documentation would be
    appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be
    misrepresented as being the original software.

    3. This notice may not be removed or altered from any source
    distribution.
*/

// c++11 required

#ifndef THREADPOOL_H_INCLUDE
#define THREADPOOL_H_INCLUDE


#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <sstream>
#include <iostream>

class ThreadPool {
public:
    ThreadPool() = default;

    ThreadPool(size_t t);

    template<typename F, typename... Args>
    std::future<typename std::result_of<F(Args...)>::type>
    add_task(F&& f, Args&&... args);

    ~ThreadPool();

    void add_thread(std::function<void()> initializer);

    void initialize(size_t t);

    void quit_all();

    void wake_up();

    void dump_status();

    void idle();

    int get_threads() const;

private:
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    std::vector<std::thread> m_threads;
    std::queue<std::function<void()>> m_tasks;
    
    std::mutex m_mutex;
    std::condition_variable m_cv;

    std::atomic<int> m_fork_threads{0};
    std::atomic<bool> m_quit{false};
    std::atomic<bool> m_idle{false};
};

inline ThreadPool::ThreadPool(size_t t) {
    initialize(t);
}

inline void ThreadPool::initialize(size_t threads) {
    for (size_t i = 0; i < threads; i++) {
        add_thread([](){} /* null function */);
    }
}

inline void ThreadPool::wake_up() {
    m_idle.store(false);
    m_cv.notify_all();
}

inline void ThreadPool::idle() {
    m_idle.store(true);
}

inline void  ThreadPool::dump_status() {
    idle();
    std::cout << "Thread pool status"                           << std::endl;
    std::cout << " Running : "         << !m_quit.load()        << std::endl;
    std::cout << " Number threads : "  << m_fork_threads.load() << std::endl;
    std::cout << " Remainning tasks: " << m_tasks.size()        << std::endl;
    wake_up();
}


inline void ThreadPool::add_thread(std::function<void()> initializer) {
    m_threads.emplace_back( [this, initializer]() -> void {

        m_fork_threads++;
        initializer();

        while(true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(this->m_mutex);
                this->m_cv.wait(lock,
                    [this](){ return this->m_quit.load() || (!this->m_tasks.empty() && !this->m_idle.load()); });

                if (this->m_quit.load()) {
                    return;
                }

                if (this->m_idle.load() || this->m_tasks.empty()) {
                    continue;
                }

                task = std::move(this->m_tasks.front());
                this->m_tasks.pop();
            }
            task();
        }
    });
}

template<typename F, typename... Args>
std::future<typename std::result_of<F(Args...)>::type>
ThreadPool::add_task(F&& f, Args&&... args) {

    const auto lambda_except = [this]() -> void {
        if (this->m_fork_threads.load() <= 0 || this->m_quit.load()) {
            auto out = std::ostringstream{};
            out << "Do not allow to add a task : ";

            if (this->m_quit.load()) {
                out << "Thread pool has stopped";
            }
            else if (this->m_fork_threads.load() <= 0) {
                out << "No threads";
            }

            throw std::runtime_error(out.str());
        }
    };

    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        lambda_except();
        m_tasks.emplace([task](){ (*task)(); });
    }
    m_cv.notify_one();

    return res;
}

inline int ThreadPool::get_threads() const {
    return m_fork_threads.load();
}

inline void ThreadPool::quit_all() {

    if (m_quit.load()) {
        return;
    }

    m_quit.store(true);
    wake_up();

    for(auto &t: m_threads) {
        m_fork_threads--;
        t.join();
    }

    while(!m_tasks.empty()) {
        m_tasks.pop();
    }
}

inline ThreadPool::~ThreadPool() {
    quit_all();
}


template<typename T>
class ThreadGroup {
public:
    ThreadGroup(ThreadPool & pool) : m_pool(pool) {}

    template<class F, class... Args>
    void add_task(F&& f, Args&&... args) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_taskresults.emplace_back(
            m_pool.add_task(std::forward<F>(f), std::forward<Args>(args)...)
        );
    }

    template<class F, class... Args>
    void fill_tasks(F&& f, Args&&... args) {
        const auto threads = m_pool.get_threads();
        for (int i = 0; i < threads; ++i) {
            add_task(std::forward<F>(f), std::forward<Args>(args)...);
        }
    }

    void wait_all() {
        for (auto && result : m_taskresults) {
            result.get();
        }
        m_taskresults.clear();
    }

    void dump_all() {
        int i = 0;
        for (auto && result : m_taskresults) {
            const auto res = result.get();
            std::cout << "[" << i++ << "] "
                      << res << std::endl;
        }
        std::cout << " end " << std::endl;
        m_taskresults.clear();
    }

private:
    std::mutex m_mutex;
    ThreadPool & m_pool;
    std::vector<std::future<T>> m_taskresults;
};

#endif
