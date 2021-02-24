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

#ifndef CPUBACKEND_H_INCLUDE
#define CPUBACKEND_H_INCLUDE
#include "Model.h"
#include "config.h"
#include "Blas.h"

class CPUBackend : public Model::NNPipe {
public:
    virtual void initialize(std::shared_ptr<Model::NNWeights> weights);
    virtual void forward(const std::vector<float> &planes,
                         const std::vector<float> &features,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_val);

    virtual void reload(std::shared_ptr<Model::NNWeights> weights);
    virtual void release();
    virtual void destroy();
    virtual bool valid();

private:
    std::shared_ptr<Model::NNWeights> m_weights{nullptr};

};

#endif
