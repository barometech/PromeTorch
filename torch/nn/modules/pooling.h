#pragma once

#include "torch/nn/module.h"
#include "torch/csrc/autograd/autograd_meta.h"
#include "torch/csrc/autograd/node.h"
#include "torch/csrc/autograd/functions/ConvBackward.h"
#ifdef PT_USE_CUDA
#include "aten/src/ATen/cuda/CUDADispatch.h"
#ifdef PT_USE_CUDNN
#include "aten/src/ATen/cudnn/CuDNN.h"
#endif
#endif
#include <array>
#include <cmath>
#include <limits>

namespace torch {
namespace nn {

// ============================================================================
// MaxPool1d
// ============================================================================

class MaxPool1d : public Module {
public:
    MaxPool1d(
        int64_t kernel_size,
        int64_t stride = 0,
        int64_t padding = 0,
        int64_t dilation = 1,
        bool ceil_mode = false
    )
        : Module("MaxPool1d")
        , kernel_size_(kernel_size)
        , stride_(stride == 0 ? kernel_size : stride)
        , padding_(padding)
        , dilation_(dilation)
        , ceil_mode_(ceil_mode)
    {}

    Tensor forward(const Tensor& input) override {
        // Input: [N, C, L]
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t in_length = input.size(2);

        int64_t out_length;
        if (ceil_mode_) {
            out_length = static_cast<int64_t>(std::ceil(
                (in_length + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1.0) / stride_ + 1));
        } else {
            out_length = (in_length + 2 * padding_ - dilation_ * (kernel_size_ - 1) - 1) / stride_ + 1;
        }

        Tensor output = at::empty({batch_size, channels, out_length});
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        // omp removed for LCC
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t ol = 0; ol < out_length; ++ol) {
                    float max_val = -std::numeric_limits<float>::infinity();

                    for (int64_t k = 0; k < kernel_size_; ++k) {
                        int64_t il = ol * stride_ - padding_ + k * dilation_;
                        if (il >= 0 && il < in_length) {
                            int64_t in_idx = n * channels * in_length + c * in_length + il;
                            max_val = std::max(max_val, in_data[in_idx]);
                        }
                    }

                    int64_t out_idx = n * channels * out_length + c * out_length + ol;
                    out_data[out_idx] = max_val;
                }
            }
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "kernel_size=" << kernel_size_ << ", stride=" << stride_;
        if (padding_ != 0) ss << ", padding=" << padding_;
        return ss.str();
    }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    int64_t dilation_;
    bool ceil_mode_;
};

// ============================================================================
// MaxPool2d
// ============================================================================

class MaxPool2d : public Module {
public:
    MaxPool2d(
        int64_t kernel_size,
        int64_t stride = 0,
        int64_t padding = 0,
        int64_t dilation = 1,
        bool ceil_mode = false
    )
        : Module("MaxPool2d")
        , kernel_size_({kernel_size, kernel_size})
        , stride_({stride == 0 ? kernel_size : stride, stride == 0 ? kernel_size : stride})
        , padding_({padding, padding})
        , dilation_({dilation, dilation})
        , ceil_mode_(ceil_mode)
    {}

    Tensor forward(const Tensor& input) override {
        // Input: [N, C, H, W]
#ifdef PT_USE_CUDA
        // CUDA dispatch (only for dilation=1) — prefer cuDNN, else custom kernel.
        if (input.is_cuda() && dilation_[0] == 1 && dilation_[1] == 1) {
#ifdef PT_USE_CUDNN
            if (input.dtype() == c10::ScalarType::Float) {
                return at::cudnn::cudnn_max_pool2d_forward(
                    input,
                    kernel_size_[0], kernel_size_[1],
                    stride_[0], stride_[1],
                    padding_[0], padding_[1]);
            }
#endif
            return at::cuda_ops::max_pool2d_forward(
                input,
                static_cast<int>(kernel_size_[0]), static_cast<int>(kernel_size_[1]),
                static_cast<int>(stride_[0]), static_cast<int>(stride_[1]),
                static_cast<int>(padding_[0]), static_cast<int>(padding_[1])
            );
        }
#endif

        // CPU implementation
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t in_height = input.size(2);
        int64_t in_width = input.size(3);

        int64_t out_height = (in_height + 2 * padding_[0] - dilation_[0] * (kernel_size_[0] - 1) - 1) / stride_[0] + 1;
        int64_t out_width = (in_width + 2 * padding_[1] - dilation_[1] * (kernel_size_[1] - 1) - 1) / stride_[1] + 1;

        Tensor output = at::empty({batch_size, channels, out_height, out_width});
        // Save argmax indices for backward (stored as float for tensor compatibility)
        bool needs_grad = autograd::GradMode::is_enabled() && input.requires_grad();
        Tensor indices;
        float* idx_data = nullptr;
        if (needs_grad) {
            indices = at::empty({batch_size, channels, out_height, out_width});
            idx_data = indices.mutable_data_ptr<float>();
        }

        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        // omp removed for LCC
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_height; ++oh) {
                    for (int64_t ow = 0; ow < out_width; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();
                        int64_t max_spatial_idx = 0;

                        for (int64_t kh = 0; kh < kernel_size_[0]; ++kh) {
                            for (int64_t kw = 0; kw < kernel_size_[1]; ++kw) {
                                int64_t ih = oh * stride_[0] - padding_[0] + kh * dilation_[0];
                                int64_t iw = ow * stride_[1] - padding_[1] + kw * dilation_[1];

                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int64_t in_idx = n * channels * in_height * in_width +
                                                    c * in_height * in_width +
                                                    ih * in_width + iw;
                                    if (in_data[in_idx] > max_val) {
                                        max_val = in_data[in_idx];
                                        max_spatial_idx = ih * in_width + iw;
                                    }
                                }
                            }
                        }

                        int64_t out_idx = n * channels * out_height * out_width +
                                         c * out_height * out_width +
                                         oh * out_width + ow;
                        out_data[out_idx] = max_val;
                        if (idx_data) {
                            idx_data[out_idx] = static_cast<float>(max_spatial_idx);
                        }
                    }
                }
            }
        }

        // Wire autograd backward
        if (needs_grad) {
            auto grad_fn = std::make_shared<autograd::MaxPool2dBackward>(
                indices, batch_size, channels, in_height, in_width
            );
            grad_fn->add_input_metadata(input);
            autograd::set_grad_fn(output, grad_fn);
            output.set_requires_grad(true);
        }

        return output;
    }

    std::string extra_repr() const override {
        std::ostringstream ss;
        ss << "kernel_size=(" << kernel_size_[0] << ", " << kernel_size_[1] << ")"
           << ", stride=(" << stride_[0] << ", " << stride_[1] << ")";
        return ss.str();
    }

    // Accessors (ONNX export)
    const std::array<int64_t, 2>& kernel_size() const { return kernel_size_; }
    const std::array<int64_t, 2>& stride() const { return stride_; }
    const std::array<int64_t, 2>& padding() const { return padding_; }
    const std::array<int64_t, 2>& dilation() const { return dilation_; }
    bool ceil_mode() const { return ceil_mode_; }

private:
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
    std::array<int64_t, 2> dilation_;
    bool ceil_mode_;
};

// ============================================================================
// AvgPool1d
// ============================================================================

class AvgPool1d : public Module {
public:
    AvgPool1d(
        int64_t kernel_size,
        int64_t stride = 0,
        int64_t padding = 0,
        bool ceil_mode = false,
        bool count_include_pad = true
    )
        : Module("AvgPool1d")
        , kernel_size_(kernel_size)
        , stride_(stride == 0 ? kernel_size : stride)
        , padding_(padding)
        , ceil_mode_(ceil_mode)
        , count_include_pad_(count_include_pad)
    {}

    Tensor forward(const Tensor& input) override {
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t in_length = input.size(2);

        int64_t out_length = (in_length + 2 * padding_ - kernel_size_) / stride_ + 1;

        Tensor output = at::zeros({batch_size, channels, out_length});
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t ol = 0; ol < out_length; ++ol) {
                    float sum = 0.0f;
                    int64_t count = 0;

                    for (int64_t k = 0; k < kernel_size_; ++k) {
                        int64_t il = ol * stride_ - padding_ + k;
                        if (il >= 0 && il < in_length) {
                            sum += in_data[n * channels * in_length + c * in_length + il];
                            count++;
                        } else if (count_include_pad_) {
                            count++;
                        }
                    }

                    out_data[n * channels * out_length + c * out_length + ol] =
                        sum / static_cast<float>(count_include_pad_ ? kernel_size_ : std::max(count, int64_t(1)));
                }
            }
        }

        return output;
    }

private:
    int64_t kernel_size_;
    int64_t stride_;
    int64_t padding_;
    bool ceil_mode_;
    bool count_include_pad_;
};

// ============================================================================
// AvgPool2d
// ============================================================================

class AvgPool2d : public Module {
public:
    AvgPool2d(
        int64_t kernel_size,
        int64_t stride = 0,
        int64_t padding = 0,
        bool ceil_mode = false,
        bool count_include_pad = true
    )
        : Module("AvgPool2d")
        , kernel_size_({kernel_size, kernel_size})
        , stride_({stride == 0 ? kernel_size : stride, stride == 0 ? kernel_size : stride})
        , padding_({padding, padding})
        , ceil_mode_(ceil_mode)
        , count_include_pad_(count_include_pad)
    {}

    Tensor forward(const Tensor& input) override {
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t in_height = input.size(2);
        int64_t in_width = input.size(3);

        int64_t out_height = (in_height + 2 * padding_[0] - kernel_size_[0]) / stride_[0] + 1;
        int64_t out_width = (in_width + 2 * padding_[1] - kernel_size_[1]) / stride_[1] + 1;

        Tensor output = at::zeros({batch_size, channels, out_height, out_width});
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        int64_t pool_size = kernel_size_[0] * kernel_size_[1];

        // omp removed for LCC
        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_height; ++oh) {
                    for (int64_t ow = 0; ow < out_width; ++ow) {
                        float sum = 0.0f;
                        int64_t count = 0;

                        for (int64_t kh = 0; kh < kernel_size_[0]; ++kh) {
                            for (int64_t kw = 0; kw < kernel_size_[1]; ++kw) {
                                int64_t ih = oh * stride_[0] - padding_[0] + kh;
                                int64_t iw = ow * stride_[1] - padding_[1] + kw;

                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    sum += in_data[n * channels * in_height * in_width +
                                                  c * in_height * in_width +
                                                  ih * in_width + iw];
                                    count++;
                                }
                            }
                        }

                        int64_t divisor = count_include_pad_ ? pool_size : std::max(count, int64_t(1));
                        out_data[n * channels * out_height * out_width +
                                c * out_height * out_width +
                                oh * out_width + ow] = sum / static_cast<float>(divisor);
                    }
                }
            }
        }

        // Wire autograd backward
        if (autograd::GradMode::is_enabled() && input.requires_grad()) {
            auto grad_fn = std::make_shared<autograd::AvgPool2dBackward>(
                batch_size, channels, in_height, in_width,
                kernel_size_, stride_, padding_, count_include_pad_
            );
            grad_fn->add_input_metadata(input);
            autograd::set_grad_fn(output, grad_fn);
            output.set_requires_grad(true);
        }

        return output;
    }

    // Accessors (ONNX export)
    const std::array<int64_t, 2>& kernel_size() const { return kernel_size_; }
    const std::array<int64_t, 2>& stride() const { return stride_; }
    const std::array<int64_t, 2>& padding() const { return padding_; }
    bool ceil_mode() const { return ceil_mode_; }
    bool count_include_pad() const { return count_include_pad_; }

private:
    std::array<int64_t, 2> kernel_size_;
    std::array<int64_t, 2> stride_;
    std::array<int64_t, 2> padding_;
    bool ceil_mode_;
    bool count_include_pad_;
};

// ============================================================================
// AdaptiveAvgPool1d
// ============================================================================

class AdaptiveAvgPool1d : public Module {
public:
    explicit AdaptiveAvgPool1d(int64_t output_size)
        : Module("AdaptiveAvgPool1d"), output_size_(output_size) {}

    Tensor forward(const Tensor& input) override {
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t in_length = input.size(2);

        Tensor output = at::zeros({batch_size, channels, output_size_});
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t ol = 0; ol < output_size_; ++ol) {
                    // Compute input range for this output
                    int64_t start = (ol * in_length) / output_size_;
                    int64_t end = ((ol + 1) * in_length) / output_size_;

                    float sum = 0.0f;
                    for (int64_t il = start; il < end; ++il) {
                        sum += in_data[n * channels * in_length + c * in_length + il];
                    }

                    out_data[n * channels * output_size_ + c * output_size_ + ol] =
                        sum / static_cast<float>(end - start);
                }
            }
        }

        return output;
    }

private:
    int64_t output_size_;
};

// ============================================================================
// AdaptiveAvgPool2d
// ============================================================================

class AdaptiveAvgPool2d : public Module {
public:
    AdaptiveAvgPool2d(int64_t output_size)
        : Module("AdaptiveAvgPool2d")
        , output_size_({output_size, output_size}) {}

    AdaptiveAvgPool2d(std::array<int64_t, 2> output_size)
        : Module("AdaptiveAvgPool2d")
        , output_size_(output_size) {}

    Tensor forward(const Tensor& input) override {
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t in_height = input.size(2);
        int64_t in_width = input.size(3);

        Tensor output = at::zeros({batch_size, channels, output_size_[0], output_size_[1]});
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < output_size_[0]; ++oh) {
                    for (int64_t ow = 0; ow < output_size_[1]; ++ow) {
                        int64_t h_start = (oh * in_height) / output_size_[0];
                        int64_t h_end = ((oh + 1) * in_height) / output_size_[0];
                        int64_t w_start = (ow * in_width) / output_size_[1];
                        int64_t w_end = ((ow + 1) * in_width) / output_size_[1];

                        float sum = 0.0f;
                        int64_t count = 0;

                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                sum += in_data[n * channels * in_height * in_width +
                                              c * in_height * in_width +
                                              ih * in_width + iw];
                                count++;
                            }
                        }

                        out_data[n * channels * output_size_[0] * output_size_[1] +
                                c * output_size_[0] * output_size_[1] +
                                oh * output_size_[1] + ow] = sum / static_cast<float>(count);
                    }
                }
            }
        }

        return output;
    }

private:
    std::array<int64_t, 2> output_size_;
};

// ============================================================================
// AdaptiveMaxPool2d
// ============================================================================

class AdaptiveMaxPool2d : public Module {
public:
    AdaptiveMaxPool2d(int64_t output_size)
        : Module("AdaptiveMaxPool2d")
        , output_size_({output_size, output_size}) {}

    Tensor forward(const Tensor& input) override {
        int64_t batch_size = input.size(0);
        int64_t channels = input.size(1);
        int64_t in_height = input.size(2);
        int64_t in_width = input.size(3);

        Tensor output = at::empty({batch_size, channels, output_size_[0], output_size_[1]});
        const float* in_data = input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        for (int64_t n = 0; n < batch_size; ++n) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < output_size_[0]; ++oh) {
                    for (int64_t ow = 0; ow < output_size_[1]; ++ow) {
                        int64_t h_start = (oh * in_height) / output_size_[0];
                        int64_t h_end = ((oh + 1) * in_height) / output_size_[0];
                        int64_t w_start = (ow * in_width) / output_size_[1];
                        int64_t w_end = ((ow + 1) * in_width) / output_size_[1];

                        float max_val = -std::numeric_limits<float>::infinity();

                        for (int64_t ih = h_start; ih < h_end; ++ih) {
                            for (int64_t iw = w_start; iw < w_end; ++iw) {
                                float val = in_data[n * channels * in_height * in_width +
                                                   c * in_height * in_width +
                                                   ih * in_width + iw];
                                max_val = std::max(max_val, val);
                            }
                        }

                        out_data[n * channels * output_size_[0] * output_size_[1] +
                                c * output_size_[0] * output_size_[1] +
                                oh * output_size_[1] + ow] = max_val;
                    }
                }
            }
        }

        return output;
    }

private:
    std::array<int64_t, 2> output_size_;
};

// ============================================================================
// GlobalAvgPool (convenience alias)
// ============================================================================

using GlobalAvgPool1d = AdaptiveAvgPool1d;

class GlobalAvgPool2d : public AdaptiveAvgPool2d {
public:
    GlobalAvgPool2d() : AdaptiveAvgPool2d(1) {}
};

} // namespace nn
} // namespace torch
