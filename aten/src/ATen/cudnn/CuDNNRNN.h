#pragma once

// ============================================================================
// cuDNN RNN / LSTM / GRU Operations
// ============================================================================
// High-performance recurrent network wrappers around the cuDNN RNN API.
//
// Supports: RNN_TANH, RNN_RELU, LSTM, GRU.
// Input layout:  x  [seq_len, batch, input_size]       (time-major)
// Output layout: y  [seq_len, batch, hidden * dirs]
// Hidden layout: hx [num_layers * dirs, batch, hidden]
// Cell  layout:  cx [num_layers * dirs, batch, hidden]  (LSTM only)
//
// Implementation notes:
// - Uses legacy cudnnRNN* API (cuDNN 7..9 all support it). The newer
//   cudnnRNNForward API requires cuDNN 8+; keeping legacy path keeps
//   us portable.
// - Weights are a single flat buffer as required by cuDNN. The helper
//   function cudnn_rnn_flat_params_size() reports the byte size. We do
//   NOT redistribute per-gate pieces here — callers (torch::nn::LSTM)
//   own the flat blob and copy into it from their module parameters
//   using cudnnGetRNNLinLayerMatrixParams when they want PyTorch-style
//   separate weights, OR (preferred here) they just own the flat blob
//   directly and treat it as the single learnable parameter.
// - Reserve space is produced by forward-training and consumed by
//   backward; caller must keep it alive between the two calls.
//
// Thread safety: handle is thread_local via CuDNNHandle. Workspace is
// a process-wide singleton (WorkspaceManager) — do not call forward
// concurrently on the same handle.
// ============================================================================

#ifdef PT_USE_CUDA
#ifdef PT_USE_CUDNN

#include "aten/src/ATen/cudnn/CuDNNHandle.h"
#include "aten/src/ATen/cudnn/CuDNNConvolution.h"   // WorkspaceManager
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"

#include <cudnn.h>
#include <cuda_runtime.h>

#include <tuple>
#include <vector>
#include <cstring>

namespace at {
namespace cudnn {

// ============================================================================
// Config
// ============================================================================

struct CuDNNRNNConfig {
    enum class RNNType { RNN_TANH, RNN_RELU, LSTM, GRU } type = RNNType::LSTM;
    int64_t input_size  = 0;
    int64_t hidden_size = 0;
    int64_t num_layers  = 1;
    bool    bidirectional = false;
    float   dropout = 0.0f;

    int num_directions() const { return bidirectional ? 2 : 1; }
    bool has_cell() const { return type == RNNType::LSTM; }
};

// ============================================================================
// RNN Descriptor RAII wrappers
// ============================================================================

class RNNDescriptor {
public:
    RNNDescriptor() : desc_(nullptr) {
        CUDNN_CHECK(cudnnCreateRNNDescriptor(&desc_));
    }
    ~RNNDescriptor() { if (desc_) cudnnDestroyRNNDescriptor(desc_); }

    cudnnRNNDescriptor_t get() const { return desc_; }
    operator cudnnRNNDescriptor_t() const { return desc_; }

    RNNDescriptor(const RNNDescriptor&) = delete;
    RNNDescriptor& operator=(const RNNDescriptor&) = delete;

private:
    cudnnRNNDescriptor_t desc_;
};

class DropoutDescriptor {
public:
    DropoutDescriptor() : desc_(nullptr), states_(nullptr), states_size_(0) {
        CUDNN_CHECK(cudnnCreateDropoutDescriptor(&desc_));
    }
    ~DropoutDescriptor() {
        if (desc_)   cudnnDestroyDropoutDescriptor(desc_);
        if (states_) cudaFree(states_);
    }

    void set(cudnnHandle_t handle, float dropout, unsigned long long seed = 0ULL) {
        CUDNN_CHECK(cudnnDropoutGetStatesSize(handle, &states_size_));
        if (states_size_ > 0) {
            cudaMalloc(&states_, states_size_);
        }
        CUDNN_CHECK(cudnnSetDropoutDescriptor(
            desc_, handle, dropout, states_, states_size_, seed));
    }

    cudnnDropoutDescriptor_t get() const { return desc_; }
    operator cudnnDropoutDescriptor_t() const { return desc_; }

    DropoutDescriptor(const DropoutDescriptor&) = delete;
    DropoutDescriptor& operator=(const DropoutDescriptor&) = delete;

private:
    cudnnDropoutDescriptor_t desc_;
    void*  states_;
    size_t states_size_;
};

// ============================================================================
// Internal helpers
// ============================================================================

inline cudnnRNNMode_t _cudnn_rnn_mode(CuDNNRNNConfig::RNNType t) {
    switch (t) {
        case CuDNNRNNConfig::RNNType::RNN_TANH: return CUDNN_RNN_TANH;
        case CuDNNRNNConfig::RNNType::RNN_RELU: return CUDNN_RNN_RELU;
        case CuDNNRNNConfig::RNNType::LSTM:     return CUDNN_LSTM;
        case CuDNNRNNConfig::RNNType::GRU:      return CUDNN_GRU;
    }
    return CUDNN_LSTM;
}

// Build per-timestep tensor descriptor array for legacy cudnnRNN* API.
// Each time step uses the same shape [batch, input_or_hidden, 1].
struct _SeqTensorDescs {
    std::vector<cudnnTensorDescriptor_t> descs;
    explicit _SeqTensorDescs(int seq_len) : descs(seq_len, nullptr) {
        for (int i = 0; i < seq_len; ++i) {
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&descs[i]));
        }
    }
    ~_SeqTensorDescs() {
        for (auto d : descs) if (d) cudnnDestroyTensorDescriptor(d);
    }
    void set_all(cudnnDataType_t dt, int batch, int features) {
        int dims[3]    = { batch, features, 1 };
        int strides[3] = { features, 1, 1 };
        for (auto d : descs) {
            CUDNN_CHECK(cudnnSetTensorNdDescriptor(d, dt, 3, dims, strides));
        }
    }
    const cudnnTensorDescriptor_t* data() const { return descs.data(); }
    _SeqTensorDescs(const _SeqTensorDescs&) = delete;
    _SeqTensorDescs& operator=(const _SeqTensorDescs&) = delete;
};

// Configure an RNN descriptor with the given config.
// cudnnSetRNNDescriptor_v6 is supported across cuDNN 7/8/9.
inline void _setup_rnn_desc(
    cudnnHandle_t handle,
    RNNDescriptor& rnn_desc,
    DropoutDescriptor& drop_desc,
    const CuDNNRNNConfig& cfg,
    cudnnDataType_t dt)
{
    drop_desc.set(handle, cfg.dropout);
    CUDNN_CHECK(cudnnSetRNNDescriptor_v6(
        handle,
        rnn_desc.get(),
        static_cast<int>(cfg.hidden_size),
        static_cast<int>(cfg.num_layers),
        drop_desc.get(),
        CUDNN_LINEAR_INPUT,
        cfg.bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
        _cudnn_rnn_mode(cfg.type),
        CUDNN_RNN_ALGO_STANDARD,
        dt
    ));
}

// ============================================================================
// Public helpers
// ============================================================================

// Required flat-params buffer size (bytes) for this configuration.
inline size_t cudnn_rnn_flat_params_size(const CuDNNRNNConfig& cfg,
                                         c10::ScalarType dtype = c10::ScalarType::Float)
{
    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dt   = getCudnnDataType(dtype);

    RNNDescriptor rnn_desc;
    DropoutDescriptor drop_desc;
    _setup_rnn_desc(handle, rnn_desc, drop_desc, cfg, dt);

    // x descriptor for a single timestep is enough to query params size.
    TensorDescriptor x_desc;
    int x_dims[3]    = { 1, static_cast<int>(cfg.input_size), 1 };
    int x_strides[3] = { static_cast<int>(cfg.input_size), 1, 1 };
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc.get(), dt, 3, x_dims, x_strides));

    size_t params_bytes = 0;
    CUDNN_CHECK(cudnnGetRNNParamsSize(
        handle, rnn_desc.get(), x_desc.get(), &params_bytes, dt));
    return params_bytes;
}

// ============================================================================
// Forward
// ============================================================================
// Inputs:
//   input:   [seq_len, batch, input_size]      (CUDA, contiguous, float)
//   hx:      [num_layers*dirs, batch, hidden]  (CUDA)
//   cx:      LSTM only, same shape as hx; pass empty tensor for non-LSTM
//   weights: flat param blob, size == cudnn_rnn_flat_params_size(config)
//
// Returns {output, hy, cy, reserve_space}.
//   output:        [seq_len, batch, hidden*dirs]
//   hy:            [num_layers*dirs, batch, hidden]
//   cy:            LSTM only; undefined Tensor otherwise
//   reserve_space: opaque buffer needed by cudnn_rnn_backward. Empty if
//                  train=false.
inline std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_rnn_forward(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& weights,
    const CuDNNRNNConfig& cfg,
    bool train)
{
    PT_ASSERT_MSG(input.is_cuda(),   "cudnn_rnn_forward: input must be CUDA");
    PT_ASSERT_MSG(hx.is_cuda(),      "cudnn_rnn_forward: hx must be CUDA");
    PT_ASSERT_MSG(weights.is_cuda(), "cudnn_rnn_forward: weights must be CUDA");

    cudnnHandle_t   handle = CuDNNHandle::get();
    cudnnDataType_t dt     = getCudnnDataType(input.dtype());

    int seq_len = static_cast<int>(input.size(0));
    int batch   = static_cast<int>(input.size(1));
    int dirs    = cfg.num_directions();
    int hidden  = static_cast<int>(cfg.hidden_size);

    RNNDescriptor     rnn_desc;
    DropoutDescriptor drop_desc;
    _setup_rnn_desc(handle, rnn_desc, drop_desc, cfg, dt);

    // Per-step descriptors for x and y
    _SeqTensorDescs x_descs(seq_len);
    x_descs.set_all(dt, batch, static_cast<int>(cfg.input_size));
    _SeqTensorDescs y_descs(seq_len);
    y_descs.set_all(dt, batch, hidden * dirs);

    // hx/cx/hy/cy descriptor
    int h_dims[3]    = { static_cast<int>(cfg.num_layers) * dirs, batch, hidden };
    int h_strides[3] = { batch * hidden, hidden, 1 };
    TensorDescriptor hx_desc, cx_desc, hy_desc, cy_desc;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hx_desc.get(), dt, 3, h_dims, h_strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cx_desc.get(), dt, 3, h_dims, h_strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hy_desc.get(), dt, 3, h_dims, h_strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cy_desc.get(), dt, 3, h_dims, h_strides));

    // Weights: filter descriptor (ND, 3 dims with 1x1 trailing).
    size_t params_bytes = 0;
    CUDNN_CHECK(cudnnGetRNNParamsSize(
        handle, rnn_desc.get(), x_descs.data()[0], &params_bytes, dt));
    int w_dims[3] = { static_cast<int>(params_bytes / sizeof(float)), 1, 1 };
    FilterDescriptor w_desc;
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(
        w_desc.get(), dt, CUDNN_TENSOR_NCHW, 3, w_dims));

    // Allocate output tensors
    auto tensor_opts = TensorOptions()
                        .dtype(input.dtype())
                        .device(input.device());
    Tensor output = empty({seq_len, batch, hidden * dirs}, tensor_opts);
    Tensor hy     = empty({cfg.num_layers * dirs, batch, hidden}, tensor_opts);
    Tensor cy;
    if (cfg.has_cell()) {
        cy = empty({cfg.num_layers * dirs, batch, hidden}, tensor_opts);
    }

    // Workspace + reserve
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle, rnn_desc.get(), seq_len, x_descs.data(), &workspace_bytes));
    void* workspace = WorkspaceManager::instance().get(workspace_bytes);

    Tensor reserve;
    size_t reserve_bytes = 0;
    void*  reserve_ptr   = nullptr;
    if (train) {
        CUDNN_CHECK(cudnnGetRNNTrainingReserveSize(
            handle, rnn_desc.get(), seq_len, x_descs.data(), &reserve_bytes));
        // Represent as a 1-D float tensor sized to cover the reserve bytes.
        int64_t nelem = static_cast<int64_t>((reserve_bytes + sizeof(float) - 1) / sizeof(float));
        if (nelem == 0) nelem = 1;
        reserve = empty({nelem}, tensor_opts);
        reserve_ptr = reserve.mutable_data_ptr<void>();
    }

    const void* cx_ptr = cfg.has_cell() ? cx.data_ptr<void>() : nullptr;
    void*       cy_ptr = cfg.has_cell() ? cy.mutable_data_ptr<void>() : nullptr;

    if (train) {
        CUDNN_CHECK(cudnnRNNForwardTraining(
            handle,
            rnn_desc.get(),
            seq_len,
            x_descs.data(), input.data_ptr<void>(),
            hx_desc.get(),  hx.data_ptr<void>(),
            cx_desc.get(),  cx_ptr,
            w_desc.get(),   weights.data_ptr<void>(),
            y_descs.data(), output.mutable_data_ptr<void>(),
            hy_desc.get(),  hy.mutable_data_ptr<void>(),
            cy_desc.get(),  cy_ptr,
            workspace, workspace_bytes,
            reserve_ptr, reserve_bytes
        ));
    } else {
        CUDNN_CHECK(cudnnRNNForwardInference(
            handle,
            rnn_desc.get(),
            seq_len,
            x_descs.data(), input.data_ptr<void>(),
            hx_desc.get(),  hx.data_ptr<void>(),
            cx_desc.get(),  cx_ptr,
            w_desc.get(),   weights.data_ptr<void>(),
            y_descs.data(), output.mutable_data_ptr<void>(),
            hy_desc.get(),  hy.mutable_data_ptr<void>(),
            cy_desc.get(),  cy_ptr,
            workspace, workspace_bytes
        ));
    }

    return std::make_tuple(std::move(output), std::move(hy), std::move(cy), std::move(reserve));
}

// ============================================================================
// Backward
// ============================================================================
// Given grad_output and (optional) grad_hy/grad_cy, produce gradients for
// input, hx, cx, and weights.
//
// Inputs (all CUDA, contiguous):
//   input, hx, cx, weights, output, hy, cy : same shapes as used in forward
//   grad_output : [seq_len, batch, hidden*dirs]
//   grad_hy     : [num_layers*dirs, batch, hidden] (pass empty for zeros)
//   grad_cy     : LSTM only, same shape as hy       (pass empty for zeros)
//   reserve     : reserve_space returned by forward(train=true)
//
// Returns {grad_input, grad_hx, grad_cx, grad_weights}. grad_cx undefined
// for non-LSTM configs.
inline std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_rnn_backward(
    const Tensor& input,
    const Tensor& hx,
    const Tensor& cx,
    const Tensor& weights,
    const Tensor& output,
    const Tensor& grad_output,
    const Tensor& grad_hy,
    const Tensor& grad_cy,
    const Tensor& reserve,
    const CuDNNRNNConfig& cfg)
{
    PT_ASSERT_MSG(reserve.defined(), "cudnn_rnn_backward: reserve space required (run forward with train=true)");

    cudnnHandle_t   handle = CuDNNHandle::get();
    cudnnDataType_t dt     = getCudnnDataType(input.dtype());

    int seq_len = static_cast<int>(input.size(0));
    int batch   = static_cast<int>(input.size(1));
    int dirs    = cfg.num_directions();
    int hidden  = static_cast<int>(cfg.hidden_size);

    RNNDescriptor     rnn_desc;
    DropoutDescriptor drop_desc;
    _setup_rnn_desc(handle, rnn_desc, drop_desc, cfg, dt);

    _SeqTensorDescs x_descs(seq_len);
    x_descs.set_all(dt, batch, static_cast<int>(cfg.input_size));
    _SeqTensorDescs y_descs(seq_len);
    y_descs.set_all(dt, batch, hidden * dirs);
    _SeqTensorDescs dx_descs(seq_len);
    dx_descs.set_all(dt, batch, static_cast<int>(cfg.input_size));
    _SeqTensorDescs dy_descs(seq_len);
    dy_descs.set_all(dt, batch, hidden * dirs);

    int h_dims[3]    = { static_cast<int>(cfg.num_layers) * dirs, batch, hidden };
    int h_strides[3] = { batch * hidden, hidden, 1 };
    TensorDescriptor hx_desc, cx_desc, dhx_desc, dcx_desc, dhy_desc, dcy_desc;
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hx_desc.get(),  dt, 3, h_dims, h_strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cx_desc.get(),  dt, 3, h_dims, h_strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(dhx_desc.get(), dt, 3, h_dims, h_strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(dcx_desc.get(), dt, 3, h_dims, h_strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(dhy_desc.get(), dt, 3, h_dims, h_strides));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(dcy_desc.get(), dt, 3, h_dims, h_strides));

    size_t params_bytes = 0;
    CUDNN_CHECK(cudnnGetRNNParamsSize(
        handle, rnn_desc.get(), x_descs.data()[0], &params_bytes, dt));
    int w_dims[3] = { static_cast<int>(params_bytes / sizeof(float)), 1, 1 };
    FilterDescriptor w_desc, dw_desc;
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc.get(),  dt, CUDNN_TENSOR_NCHW, 3, w_dims));
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(dw_desc.get(), dt, CUDNN_TENSOR_NCHW, 3, w_dims));

    auto tensor_opts = TensorOptions()
                        .dtype(input.dtype())
                        .device(input.device());
    Tensor grad_input   = empty(input.sizes(),   tensor_opts);
    Tensor grad_hx      = empty(hx.sizes(),      tensor_opts);
    Tensor grad_cx;
    if (cfg.has_cell()) grad_cx = empty(hx.sizes(), tensor_opts);
    Tensor grad_weights = empty(weights.sizes(), tensor_opts);
    // cuDNN accumulates into dw; zero it.
    cudaMemset(grad_weights.mutable_data_ptr<void>(), 0,
               grad_weights.numel() * sizeof(float));

    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(
        handle, rnn_desc.get(), seq_len, x_descs.data(), &workspace_bytes));
    void* workspace = WorkspaceManager::instance().get(workspace_bytes);

    size_t reserve_bytes = reserve.numel() * sizeof(float);
    void*  reserve_ptr   = reserve.mutable_data_ptr<void>();

    const void* cx_ptr      = cfg.has_cell() ? cx.data_ptr<void>()           : nullptr;
    const void* grad_cy_ptr = (cfg.has_cell() && grad_cy.defined()) ? grad_cy.data_ptr<void>() : nullptr;
    const void* grad_hy_ptr = grad_hy.defined() ? grad_hy.data_ptr<void>() : nullptr;
    void*       grad_cx_ptr = cfg.has_cell() ? grad_cx.mutable_data_ptr<void>() : nullptr;

    // ---- backward data (produces dx, dhx, dcx) ----
    CUDNN_CHECK(cudnnRNNBackwardData(
        handle,
        rnn_desc.get(),
        seq_len,
        y_descs.data(),  output.data_ptr<void>(),
        dy_descs.data(), grad_output.data_ptr<void>(),
        dhy_desc.get(),  grad_hy_ptr,
        dcy_desc.get(),  grad_cy_ptr,
        w_desc.get(),    weights.data_ptr<void>(),
        hx_desc.get(),   hx.data_ptr<void>(),
        cx_desc.get(),   cx_ptr,
        dx_descs.data(), grad_input.mutable_data_ptr<void>(),
        dhx_desc.get(),  grad_hx.mutable_data_ptr<void>(),
        dcx_desc.get(),  grad_cx_ptr,
        workspace, workspace_bytes,
        reserve_ptr, reserve_bytes
    ));

    // ---- backward weights (accumulates into dw) ----
    CUDNN_CHECK(cudnnRNNBackwardWeights(
        handle,
        rnn_desc.get(),
        seq_len,
        x_descs.data(), input.data_ptr<void>(),
        hx_desc.get(),  hx.data_ptr<void>(),
        y_descs.data(), output.data_ptr<void>(),
        workspace, workspace_bytes,
        dw_desc.get(),  grad_weights.mutable_data_ptr<void>(),
        reserve_ptr, reserve_bytes
    ));

    return std::make_tuple(std::move(grad_input), std::move(grad_hx),
                           std::move(grad_cx),    std::move(grad_weights));
}

// ============================================================================
// Weight packer — copy PyTorch-style per-gate weights into cuDNN's flat blob
// ============================================================================
// For a given layer/direction/linear_layer_id index, returns a pair
// {ptr, nelem} pointing inside the flat params blob. The caller then
// cudaMemcpy's its own source weight/bias into that slot.
//
// cuDNN linear layer indices (matches docs):
//   RNN_TANH / RNN_RELU : 0 = W_ih, 1 = W_hh      (bias: 0 = b_ih, 1 = b_hh)
//   LSTM                : 0..3 = W_ih[i,f,g,o], 4..7 = W_hh[i,f,g,o]
//   GRU                 : 0..2 = W_ih[r,z,n],   3..5 = W_hh[r,z,n]
// (plus corresponding biases at the same indices through the *bias* query).
struct RNNLinLayerSlot {
    float* ptr;
    int64_t nelem;
};

inline RNNLinLayerSlot cudnn_rnn_get_lin_layer_matrix(
    const Tensor& flat_weights,
    const CuDNNRNNConfig& cfg,
    int layer,
    int direction,
    int lin_layer_id)
{
    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dt   = getCudnnDataType(flat_weights.dtype());

    RNNDescriptor rnn_desc;
    DropoutDescriptor drop_desc;
    _setup_rnn_desc(handle, rnn_desc, drop_desc, cfg, dt);

    TensorDescriptor x_desc;
    int x_dims[3]    = { 1, static_cast<int>(cfg.input_size), 1 };
    int x_strides[3] = { static_cast<int>(cfg.input_size), 1, 1 };
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc.get(), dt, 3, x_dims, x_strides));

    size_t params_bytes = 0;
    CUDNN_CHECK(cudnnGetRNNParamsSize(handle, rnn_desc.get(), x_desc.get(), &params_bytes, dt));
    int w_dims[3] = { static_cast<int>(params_bytes / sizeof(float)), 1, 1 };
    FilterDescriptor w_desc;
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc.get(), dt, CUDNN_TENSOR_NCHW, 3, w_dims));

    int pseudo = layer * cfg.num_directions() + direction;

    FilterDescriptor lin_desc;
    void* lin_ptr = nullptr;
    CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
        handle, rnn_desc.get(),
        pseudo,
        x_desc.get(),
        w_desc.get(), flat_weights.data_ptr<void>(),
        lin_layer_id,
        lin_desc.get(),
        &lin_ptr
    ));

    cudnnDataType_t fdt;
    cudnnTensorFormat_t fmt;
    int nbd = 0;
    int fdims[8] = {0};
    CUDNN_CHECK(cudnnGetFilterNdDescriptor(lin_desc.get(), 3, &fdt, &fmt, &nbd, fdims));
    int64_t nelem = 1;
    for (int i = 0; i < nbd; ++i) nelem *= fdims[i];
    return { static_cast<float*>(lin_ptr), nelem };
}

inline RNNLinLayerSlot cudnn_rnn_get_lin_layer_bias(
    const Tensor& flat_weights,
    const CuDNNRNNConfig& cfg,
    int layer,
    int direction,
    int lin_layer_id)
{
    cudnnHandle_t handle = CuDNNHandle::get();
    cudnnDataType_t dt   = getCudnnDataType(flat_weights.dtype());

    RNNDescriptor rnn_desc;
    DropoutDescriptor drop_desc;
    _setup_rnn_desc(handle, rnn_desc, drop_desc, cfg, dt);

    TensorDescriptor x_desc;
    int x_dims[3]    = { 1, static_cast<int>(cfg.input_size), 1 };
    int x_strides[3] = { static_cast<int>(cfg.input_size), 1, 1 };
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc.get(), dt, 3, x_dims, x_strides));

    size_t params_bytes = 0;
    CUDNN_CHECK(cudnnGetRNNParamsSize(handle, rnn_desc.get(), x_desc.get(), &params_bytes, dt));
    int w_dims[3] = { static_cast<int>(params_bytes / sizeof(float)), 1, 1 };
    FilterDescriptor w_desc;
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc.get(), dt, CUDNN_TENSOR_NCHW, 3, w_dims));

    int pseudo = layer * cfg.num_directions() + direction;

    FilterDescriptor lin_desc;
    void* lin_ptr = nullptr;
    CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(
        handle, rnn_desc.get(),
        pseudo,
        x_desc.get(),
        w_desc.get(), flat_weights.data_ptr<void>(),
        lin_layer_id,
        lin_desc.get(),
        &lin_ptr
    ));

    cudnnDataType_t fdt;
    cudnnTensorFormat_t fmt;
    int nbd = 0;
    int fdims[8] = {0};
    CUDNN_CHECK(cudnnGetFilterNdDescriptor(lin_desc.get(), 3, &fdt, &fmt, &nbd, fdims));
    int64_t nelem = 1;
    for (int i = 0; i < nbd; ++i) nelem *= fdims[i];
    return { static_cast<float*>(lin_ptr), nelem };
}

} // namespace cudnn
} // namespace at

#endif // PT_USE_CUDNN
#endif // PT_USE_CUDA
