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

#include <numeric>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include "Board.h"
#include "Search.h"
#include "UCTNode.h"
#include "Random.h"
#include "Model.h"
#include "Decoder.h"
#include "config.h"

Search::Search(Position &position, Network &network, Train &train) : 
    m_position(position),  m_network(network), m_train(train) {

    m_parameters = std::make_shared<SearchParameters>();

    const auto t = m_parameters->threads;
    m_searchpool.initialize(t);
    m_search_group = std::make_unique<ThreadGroup<void>>(m_searchpool);

    m_maxplayouts = m_parameters->playouts;
    m_maxvisits = m_parameters->visits;
}

std::shared_ptr<SearchParameters> Search::parameters() const {
    return m_parameters;
}

Search::~Search() {
    clear_nodes();
    m_search_group->wait_all();
}

bool Search::is_running() {
    return m_running.load();
}

void Search::set_running(bool running) {
    m_running.store(running);
}

bool Search::stop_thinking(int elapsed, int limittime) const {
   return elapsed > limittime ||
              m_playouts.load() >= m_maxplayouts ||
              m_rootnode->get_visits() >= m_maxvisits;
}

void Search::increment_threads() {
    m_running_threads.fetch_add(1);
}

void Search::decrement_threads() {
    m_running_threads.fetch_sub(1);
}

void Search::think(SearchSetting setting, SearchInformation *info) {
    m_setting = setting;
    if (option<std::string>("search_mode") == "uct") {
        uct_think(info);
    } else if (option<std::string>("search_mode") == "alphabeta") {
        alpha_beta_think(info);
    }
}

void Search::interrupt() {
    set_running(false);
    m_search_group->wait_all();
}

void Search::ponderhit(bool draw) {
    std::lock_guard<std::mutex> lock(m_thinking_mtx);
    m_setting.ponder = false;
    m_setting.draw = draw;
}
