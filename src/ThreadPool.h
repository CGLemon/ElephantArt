/*
    Extended from code:
    Copyright (c) 2012 Jakob Progsch, Václav Zeman
    Copyright (c) 2017-2019 Gian-Carlo Pascutto and contributors
    Modifications:
    Copyright (c) 2020-2021 Hung Zhe Lin

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

#ifndef THREAD_H_INCLUDE
#define THREAD_H_INCLUDE

#include <atomic>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <iostream>

class ThreadPool {
public:
    ThreadPool(size_t threads);
    ~ThreadPool();

    static ThreadPool& get(size_t threads=0);

    template<class F, class... Args>
    auto add_task(F&& f, Args&&... args)
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    size_t get_num_threads() const;

private:
    void add_thread(std::function<void()> initializer);
    
    bool is_stop_running() const;

    std::atomic<bool> m_running{false};

    // Number of allocated threads.
    std::atomic<size_t> m_num_threads{0};
  
    // Need to keep track of threads so we can join them.
    std::vector<std::thread> m_workers;

    // The task queue.
    std::queue<std::function<void(void)>> m_tasks;

    std::mutex m_queue_mutex;
    
    std::condition_variable m_cv;
};

// Get the global thread pool.
inline ThreadPool& ThreadPool::get(size_t threads) {
    static ThreadPool pool(0);
    while (threads > pool.get_num_threads()) {
        pool.add_thread([](){});
    }
    while (threads < pool.get_num_threads() && threads != 0) {
        break;
    }
    return pool;
}

// The constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads) {
    m_running.store(false, std::memory_order_relaxed);
    for (size_t t = 0; t < threads ; ++t) {
        add_thread([](){});
    }

    // Wait for thread construction.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

inline void ThreadPool::add_thread(std::function<void()> initializer) {
    m_num_threads.fetch_add(1);
    m_workers.emplace_back(
        [this, initializer]() -> void {
            initializer();
            while (true) {
                auto task = std::function<void(void)>{};
                {
                    std::unique_lock<std::mutex> lock(m_queue_mutex);
                    m_cv.wait(lock,
                        [this](){ return is_stop_running() || !m_tasks.empty(); });
                    if (is_stop_running() && m_tasks.empty()) break;
                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }
                task();
            }
        }
    );
}

inline size_t ThreadPool::get_num_threads() const {
    return m_num_threads.load(std::memory_order_relaxed);
}

inline bool ThreadPool::is_stop_running() const {
    return m_running.load(std::memory_order_relaxed);
}

// Add new work item to the pool.
template<class F, class... Args>
auto ThreadPool::add_task(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::lock_guard<std::mutex> lock(m_queue_mutex);
        m_tasks.emplace([task](){ (*task)(); });
    }
    m_cv.notify_one();
    return res;
}

// The destructor joins all threads.
inline ThreadPool::~ThreadPool()
{
    m_running.store(true, std::memory_order_relaxed);
    m_cv.notify_all();
    for(auto &worker: m_workers) {
        worker.join();
    }
}

template<typename T>
class ThreadGroup {
public:
    ThreadGroup(ThreadPool *pool) {
        m_pool = pool;
    }
    ThreadGroup(ThreadGroup &&group) {
        m_pool = group.m_pool;
    }

    template<class F, class... Args>
    void add_task(F&& f, Args&&... args) {
        m_tasks_future.emplace_back(
            m_pool->add_task(std::forward<F>(f), std::forward<Args>(args)...));
    }

    void wait_to_join() {
        for (auto &&res : m_tasks_future) {
            res.get();
        }
        m_tasks_future.clear();
    }

private:
    ThreadPool *m_pool;
    std::vector<std::future<T>> m_tasks_future;
};
#endif
