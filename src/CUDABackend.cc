#ifdef USE_CUDA
#include "CUDABackend.h"
#include "cuda/CUDACommon.h"
#include "config.h"
#include "Utils.h"
#include "Model.h"
#include "Board.h"

#include <iterator>
#include <chrono>


void CUDABackend::initialize(std::shared_ptr<Model::NNWeights> weights) {

    assert(m_nnworkers.size() == 0);
    const auto cnt = CUDA::get_devicecount();

    if (option<int>("gpu") >= 0) {
        m_nnworkers.emplace_back(std::make_unique<NNWorker>());
        const auto gpu = option<int>("gpu") >= cnt ? 0 : option<int>("gpu");
        m_nnworkers[0]->build_graph(gpu, weights);
    } else {
        const auto cnt = CUDA::get_devicecount();
        for (int i = 0; i < cnt; ++i) {
            m_nnworkers.emplace_back(std::make_unique<NNWorker>());
        }

        for (int i = 0; i < cnt; ++i) {
            m_nnworkers[i]->build_graph(i, weights);
        }
    }
}


void CUDABackend::NNWorker::build_graph(const int gpu, std::shared_ptr<Model::NNWeights> weights) {

    if (m_graph != nullptr) {
        return;
    }

    m_gpu = gpu;
    auto d = CUDA::get_device(m_gpu);
    cudaSetDevice(d);
    
    m_scratch_size = 0;
    m_maxbatch = option<int>("batchsize");

    const auto output_channels = weights->residual_channels;

    // build graph

    // input layer
    m_graph->input_conv = CUDA::Convolve(
        m_maxbatch,          // max batch size
        3,                   // kernel size
        INPUT_CHANNELS,      // input channels
        output_channels      // output channels
    );
    m_graph->input_bnorm = CUDA::Batchnorm(
        m_maxbatch,          // max batch size
        output_channels      // channels
    );

    // residual tower
    const auto residuals = weights->residual_blocks;
    for (int i = 0; i < residuals; ++i) {
        m_graph->tower_conv.emplace_back(CUDA::Convolve{});
        m_graph->tower_bnorm.emplace_back(CUDA::Batchnorm{});
        m_graph->tower_conv.emplace_back(CUDA::Convolve{});
        m_graph->tower_bnorm.emplace_back(CUDA::Batchnorm{});
        m_graph->tower_se.emplace_back(CUDA::SEUnit{});
    }

    for (int i = 0; i < residuals; ++i) {
        const auto off_set = 2 * i;
        const auto tower_channels = weights->residual_channels;
        const auto tower_ptr = weights->residual_tower.data() + i;
    
        m_graph->tower_conv[off_set+0] = CUDA::Convolve(
            m_maxbatch,          // max batch size
            3,                   // kernel size
            tower_channels,      // input channels
            tower_channels       // output channels
        );
        m_graph->tower_bnorm[off_set+0] = CUDA::Batchnorm(
            m_maxbatch,          // max batch size
            tower_channels       // channels
        );

        m_graph->tower_conv[off_set+1] = CUDA::Convolve(
            m_maxbatch,          // max batch size
            3,                   // kernel size
            tower_channels,      // input channels
            tower_channels       // output channels
        );
        m_graph->tower_bnorm[off_set+1] = CUDA::Batchnorm(
            m_maxbatch,          // max batch size
            tower_channels       // channels
        );

        if (tower_ptr->apply_se) {
            const size_t se_size = tower_ptr->se_size;
            m_graph->tower_se[i] = CUDA::SEUnit(
                m_maxbatch,      // max batch size
                tower_channels,  // channels
                se_size          // SE size
            );
        }
    }

    // policy head
    const auto policy_extract_channels = weights->policy_extract_channels;
    m_graph->p_ex_conv = CUDA::Convolve(
        m_maxbatch,             // max batch size
        3,                      // kernel size
        output_channels,        // input channels
        policy_extract_channels // output channels
    );
    m_graph->p_ex_bnorm = CUDA::Batchnorm(
        m_maxbatch,             // max batch size
        policy_extract_channels // channels
    );
    m_graph->p_map = CUDA::Convolve(
        m_maxbatch,              // max batch size
        3,                       // kernel size
        policy_extract_channels, // input channels
        POLICYMAP                // output channels
    );

    // value head
    const auto value_extract_channels = weights->value_extract_channels;
    m_graph->v_ex_conv = CUDA::Convolve(
        m_maxbatch,             // max batch size
        1,                      // kernel size
        output_channels,        // input channels
        value_extract_channels  // output channels
    );
    m_graph->v_ex_bnorm = CUDA::Batchnorm(
        m_maxbatch,             // max batch size
        value_extract_channels  // channels
    );
    m_graph->v_fc1 = CUDA::FullyConnect(
        m_maxbatch,                                     // max batch size
        value_extract_channels * Board::INTERSECTIONS,  // input size
        VALUELAYER,                                     // output size
        true                                            // relu
    );
    m_graph->v_fc2 = CUDA::FullyConnect(
        m_maxbatch,     // max batch size
        VALUELAYER,     // input size
        WINRATELAYER,   // output size
        false           // relu
    );

    // fill the parameters
    // input layer
    m_graph->input_conv.LoadingWeight(weights->input_conv.weights, m_scratch_size);
    m_graph->input_bnorm.LoadingWeight(weights->input_bn.means, weights->input_bn.stddevs);

    // residual tower
    for (int i = 0; i < residuals; ++i) {
        const auto off_set = 2 * i;
        const auto tower_ptr = weights->residual_tower.data() + i;

        m_graph->tower_conv[off_set+0].LoadingWeight(tower_ptr->conv_1.weights, m_scratch_size);
        m_graph->tower_bnorm[off_set+0].LoadingWeight(tower_ptr->bn_1.means, tower_ptr->bn_1.stddevs);
        m_graph->tower_conv[off_set+1].LoadingWeight(tower_ptr->conv_2.weights, m_scratch_size);
        m_graph->tower_bnorm[off_set+1].LoadingWeight(tower_ptr->bn_2.means, tower_ptr->bn_2.stddevs);

        if (tower_ptr->apply_se) {
            m_graph->tower_se[i].LoadingWeight(tower_ptr->extend.weights,
                                               tower_ptr->extend.biases,
                                               tower_ptr->squeeze.weights,
                                               tower_ptr->squeeze.biases);
        }
    }

    // policy head
    m_graph->p_ex_conv.LoadingWeight(weights->p_ex_conv.weights, m_scratch_size);
    m_graph->p_ex_bnorm.LoadingWeight(weights->p_ex_bn.means, weights->p_ex_bn.stddevs);
    m_graph->p_map.LoadingWeight(weights->p_map.weights, weights->p_map.biases, m_scratch_size);

    // value head
    m_graph->v_ex_conv.LoadingWeight(weights->v_ex_conv.weights, m_scratch_size);
    m_graph->v_ex_bnorm.LoadingWeight(weights->v_ex_bn.means, weights->v_ex_bn.stddevs);
    m_graph->v_fc1.LoadingWeight(weights->v_fc1.weights, weights->v_fc1.biases);
    m_graph->v_fc2.LoadingWeight(weights->v_fc2.weights, weights->v_fc2.biases);

    const auto factor = m_maxbatch * sizeof(float);
    const auto inputs_size = static_cast<size_t>(factor * INPUT_CHANNELS * Board::INTERSECTIONS);
    const auto pol_size = static_cast<size_t>(factor * POLICYMAP * Board::INTERSECTIONS);
    const auto val_size = static_cast<size_t>(factor * WINRATELAYER);

    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_scratch, m_scratch_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_inputs, inputs_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_pol, pol_size));
    CUDA::ReportCUDAErrors(cudaMalloc(&cuda_val, val_size));
    cudaSetDevice(0);
}

void CUDABackend::NNWorker::destroy_graph() {
    if (m_graph == nullptr) {
        return;
    }
    CUDA::ReportCUDAErrors(cudaFree(cuda_scratch));
    CUDA::ReportCUDAErrors(cudaFree(cuda_inputs));
    CUDA::ReportCUDAErrors(cudaFree(cuda_pol));
    CUDA::ReportCUDAErrors(cudaFree(cuda_val));
    m_graph.reset();
    m_graph = nullptr;
}

CUDABackend::NNWorker::~NNWorker() {
    destroy_graph();
}
#endif           
