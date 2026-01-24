// ============================================================================
// CUDA Convolution Kernels for PromeTorch
// ============================================================================
// Direct convolution implementation for GPU
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

namespace at {
namespace cuda {

// ============================================================================
// Conv2d Forward Kernel
// ============================================================================
// Direct convolution (not im2col based - simpler and works for small kernels)
// Input: [N, C_in, H, W]
// Weight: [C_out, C_in/groups, kH, kW]
// Output: [N, C_out, H_out, W_out]

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups
) {
    // Each thread computes one output element
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_channels * out_height * out_width;

    if (out_idx >= total_outputs) return;

    // Decode output index
    int w = out_idx % out_width;
    int h = (out_idx / out_width) % out_height;
    int c_out = (out_idx / (out_width * out_height)) % out_channels;
    int n = out_idx / (out_width * out_height * out_channels);

    int group_out_channels = out_channels / groups;
    int group_in_channels = in_channels / groups;
    int g = c_out / group_out_channels;
    int c_out_in_group = c_out % group_out_channels;

    float sum = 0.0f;

    // Iterate over input channels in this group and kernel positions
    for (int c_in = 0; c_in < group_in_channels; ++c_in) {
        int actual_c_in = g * group_in_channels + c_in;

        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int ih = h * stride_h - padding_h + kh * dilation_h;
                int iw = w * stride_w - padding_w + kw * dilation_w;

                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                    int in_idx = n * in_channels * in_height * in_width +
                                actual_c_in * in_height * in_width +
                                ih * in_width + iw;

                    int w_idx = c_out * group_in_channels * kernel_h * kernel_w +
                               c_in * kernel_h * kernel_w +
                               kh * kernel_w + kw;

                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += bias[c_out];
    }

    output[out_idx] = sum;
}

// Launch conv2d forward
void launch_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    cudaStream_t stream
) {
    int total_outputs = batch_size * out_channels * out_height * out_width;
    int block_size = 256;
    int num_blocks = (total_outputs + block_size - 1) / block_size;

    conv2d_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, out_height, out_width,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w,
        groups
    );
}

// ============================================================================
// MaxPool2d Forward Kernel
// ============================================================================

__global__ void max_pool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;

    if (out_idx >= total_outputs) return;

    int w = out_idx % out_width;
    int h = (out_idx / out_width) % out_height;
    int c = (out_idx / (out_width * out_height)) % channels;
    int n = out_idx / (out_width * out_height * channels);

    float max_val = -1e38f;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = h * stride_h - padding_h + kh;
            int iw = w * stride_w - padding_w + kw;

            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                int in_idx = n * channels * in_height * in_width +
                            c * in_height * in_width +
                            ih * in_width + iw;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
    }

    output[out_idx] = max_val;
}

void launch_max_pool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    cudaStream_t stream
) {
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int num_blocks = (total_outputs + block_size - 1) / block_size;

    max_pool2d_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
        input, output,
        batch_size, channels, in_height, in_width,
        out_height, out_width,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w
    );
}

// ============================================================================
// AvgPool2d Forward Kernel
// ============================================================================

__global__ void avg_pool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    bool count_include_pad
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;

    if (out_idx >= total_outputs) return;

    int w = out_idx % out_width;
    int h = (out_idx / out_width) % out_height;
    int c = (out_idx / (out_width * out_height)) % channels;
    int n = out_idx / (out_width * out_height * channels);

    float sum = 0.0f;
    int count = 0;

    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            int ih = h * stride_h - padding_h + kh;
            int iw = w * stride_w - padding_w + kw;

            if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                int in_idx = n * channels * in_height * in_width +
                            c * in_height * in_width +
                            ih * in_width + iw;
                sum += input[in_idx];
                count++;
            } else if (count_include_pad) {
                count++;
            }
        }
    }

    if (count_include_pad) {
        count = kernel_h * kernel_w;
    }

    output[out_idx] = (count > 0) ? (sum / count) : 0.0f;
}

void launch_avg_pool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    bool count_include_pad,
    cudaStream_t stream
) {
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int num_blocks = (total_outputs + block_size - 1) / block_size;

    avg_pool2d_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
        input, output,
        batch_size, channels, in_height, in_width,
        out_height, out_width,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, count_include_pad
    );
}

// ============================================================================
// AdaptiveAvgPool2d Forward Kernel
// ============================================================================

__global__ void adaptive_avg_pool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_height * out_width;

    if (out_idx >= total_outputs) return;

    int w = out_idx % out_width;
    int h = (out_idx / out_width) % out_height;
    int c = (out_idx / (out_width * out_height)) % channels;
    int n = out_idx / (out_width * out_height * channels);

    // Compute pooling region
    int h_start = (h * in_height) / out_height;
    int h_end = ((h + 1) * in_height + out_height - 1) / out_height;
    int w_start = (w * in_width) / out_width;
    int w_end = ((w + 1) * in_width + out_width - 1) / out_width;

    float sum = 0.0f;
    int count = 0;

    for (int ih = h_start; ih < h_end; ++ih) {
        for (int iw = w_start; iw < w_end; ++iw) {
            int in_idx = n * channels * in_height * in_width +
                        c * in_height * in_width +
                        ih * in_width + iw;
            sum += input[in_idx];
            count++;
        }
    }

    output[out_idx] = (count > 0) ? (sum / count) : 0.0f;
}

void launch_adaptive_avg_pool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    cudaStream_t stream
) {
    int total_outputs = batch_size * channels * out_height * out_width;
    int block_size = 256;
    int num_blocks = (total_outputs + block_size - 1) / block_size;

    adaptive_avg_pool2d_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
        input, output,
        batch_size, channels, in_height, in_width,
        out_height, out_width
    );
}

// ============================================================================
// BatchNorm2d Forward Kernel (Inference mode)
// ============================================================================

__global__ void batch_norm2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,  // weight
    const float* __restrict__ beta,   // bias
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * height * width;

    if (idx >= total) return;

    int w = idx % width;
    int h = (idx / width) % height;
    int c = (idx / (width * height)) % channels;
    // int n = idx / (width * height * channels);  // not needed

    float mean = running_mean[c];
    float var = running_var[c];
    float g = gamma[c];
    float b = beta[c];

    float x = input[idx];
    float x_norm = (x - mean) / sqrtf(var + eps);
    output[idx] = g * x_norm + b;
}

void launch_batch_norm2d_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* running_mean,
    const float* running_var,
    float* output,
    int batch_size,
    int channels,
    int height,
    int width,
    float eps,
    cudaStream_t stream
) {
    int total = batch_size * channels * height * width;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    batch_norm2d_forward_kernel<<<num_blocks, block_size, 0, stream>>>(
        input, gamma, beta, running_mean, running_var, output,
        batch_size, channels, height, width, eps
    );
}

// ============================================================================
// Flatten / Unflatten utilities (just memory copies with reshape semantics)
// ============================================================================

// Nothing needed - reshape is just view change, no data copy

} // namespace cuda
} // namespace at
