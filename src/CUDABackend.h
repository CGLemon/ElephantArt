#ifndef CUDABACKEND_H_INCLUDE
#define CUDABACKEND_H_INCLUDE
#ifdef USE_CUDA
#include "Model.h"
#include "config.h"
#include "cuda/CUDALayers.h"
#include "cuda/CUDAKernels.h"
#include "cuda/CUDACommon.h"

#include <atomic>
#include <memory>
#include <list>
#include <array>
#include <vector>
#include <mutex>
#include <thread>
#include <condition_variable>

class CUDABackend : public Model::NNPipe {
public:
    virtual void initialize(std::shared_ptr<Model::NNWeights> weights);
    virtual void forward(const std::vector<float> &planes,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_val);

    virtual void reload(std::shared_ptr<Model::NNWeights> weights);
    virtual void release();
    virtual void destroy();
    virtual bool valid();

private:
   struct Graph {
        // intput
        CUDA::Convolve input_conv;
        CUDA::Batchnorm input_bnorm;

        // residual towers
        std::vector<CUDA::Convolve> tower_conv;
        std::vector<CUDA::Batchnorm> tower_bnorm;
        std::vector<CUDA::SEUnit> tower_se;

        // policy head 
        CUDA::Convolve p_ex_conv;
        CUDA::Batchnorm p_ex_bnorm;
        CUDA::Convolve p_map;
  
        // value head
        CUDA::Convolve v_ex_conv;
        CUDA::Batchnorm v_ex_bnorm;
        CUDA::FullyConnect v_fc1;
        CUDA::FullyConnect v_fc2;
    };

    class NNWorker {
    public:
         ~NNWorker();
        void build_graph(const int gpu, std::shared_ptr<Model::NNWeights> weights);
        void destroy_graph();
        void idle_loop();
        void quit();

    private:
        int m_gpu;
        int m_maxbatch;
        std::unique_ptr<Graph> m_graph{nullptr};
        void *cuda_scratch;

        void *cuda_inputs;
        void *cuda_pol;
        void *cuda_val;

        size_t m_scratch_size;

    };

    std::vector<std::unique_ptr<NNWorker>> m_nnworkers;
    std::shared_ptr<Model::NNWeights> m_weights{nullptr};

}; 
#endif
#endif
