#pragma once

#define _USE_MATH_DEFINES
#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace at {
namespace native {

// ============================================================================
// Internal: Cooley-Tukey FFT (radix-2, DIT)
// Works on interleaved complex arrays: [re0, im0, re1, im1, ...]
// ============================================================================

namespace fft_detail {

inline void bit_reverse(double* data, int64_t n) {
    int64_t j = 0;
    for (int64_t i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[2 * i], data[2 * j]);
            std::swap(data[2 * i + 1], data[2 * j + 1]);
        }
        int64_t m = n / 2;
        while (m >= 1 && j >= m) {
            j -= m;
            m /= 2;
        }
        j += m;
    }
}

// In-place Cooley-Tukey FFT
// inverse=false: forward FFT, inverse=true: inverse FFT
inline void cooley_tukey(double* data, int64_t n, bool inverse) {
    if (n <= 1) return;

    bit_reverse(data, n);

    for (int64_t len = 2; len <= n; len *= 2) {
        double angle = 2.0 * M_PI / len * (inverse ? -1.0 : 1.0);
        double w_re = std::cos(angle);
        double w_im = std::sin(angle);

        for (int64_t i = 0; i < n; i += len) {
            double cur_re = 1.0, cur_im = 0.0;
            for (int64_t j = 0; j < len / 2; ++j) {
                int64_t u = i + j;
                int64_t v = i + j + len / 2;

                double t_re = cur_re * data[2 * v] - cur_im * data[2 * v + 1];
                double t_im = cur_re * data[2 * v + 1] + cur_im * data[2 * v];

                data[2 * v] = data[2 * u] - t_re;
                data[2 * v + 1] = data[2 * u + 1] - t_im;
                data[2 * u] += t_re;
                data[2 * u + 1] += t_im;

                double new_re = cur_re * w_re - cur_im * w_im;
                double new_im = cur_re * w_im + cur_im * w_re;
                cur_re = new_re;
                cur_im = new_im;
            }
        }
    }

    if (inverse) {
        for (int64_t i = 0; i < 2 * n; ++i) {
            data[i] /= n;
        }
    }
}

// Pad to next power of 2
inline int64_t next_power_of_2(int64_t n) {
    int64_t p = 1;
    while (p < n) p *= 2;
    return p;
}

} // namespace fft_detail

// ============================================================================
// fft: 1D complex-to-complex FFT
// Input: [..., N, 2] (complex format), Output: [..., N, 2]
// If input is real (no last dim of 2), treat as complex with zero imaginary
// ============================================================================

inline Tensor fft(const Tensor& self, int64_t n = -1, int64_t dim = -1) {
    Tensor input = self.contiguous();
    int64_t ndim = input.dim();

    bool is_complex = (ndim >= 2 && input.size(-1) == 2);

    // Default dim: for complex [.., N, 2] → ndim-2, for real [.., N] → ndim-1
    if (dim == -1) {
        dim = is_complex ? ndim - 2 : ndim - 1;
    } else if (dim < 0) {
        dim += ndim;
    }

    // If real input, add zero imaginary part
    Tensor complex_input;
    if (!is_complex) {
        // Real tensor: create complex [..., 2]
        Tensor zeros_like = at::zeros(input.sizes());
        complex_input = at::native::stack({input, zeros_like}, -1);  // [..., 2]
        ndim = complex_input.dim();
        // dim stays the same in the leading dims
    } else {
        complex_input = input;
        // For complex input, the FFT dim is in the leading dims (not the last dim=2)
    }

    // Get the FFT length
    int64_t fft_dim = dim;
    int64_t N = complex_input.size(fft_dim);
    int64_t target_n = (n > 0) ? n : N;
    int64_t padded_n = fft_detail::next_power_of_2(target_n);

    // Create output
    auto out_shape = complex_input.sizes().vec();
    out_shape[fft_dim] = padded_n;
    Tensor output = at::zeros(out_shape);

    // For [..., N, 2] tensor where fft_dim == ndim-2: simple fast path
    if (is_complex && fft_dim == ndim - 2) {
        int64_t batch = 1;
        for (int64_t d = 0; d < ndim - 2; ++d) batch *= complex_input.size(d);

        const float* in_data = complex_input.data_ptr<float>();
        float* out_data = output.mutable_data_ptr<float>();

        for (int64_t b = 0; b < batch; ++b) {
            std::vector<double> buf(2 * padded_n, 0.0);

            // Copy input to buffer
            for (int64_t i = 0; i < std::min(N, target_n); ++i) {
                buf[2 * i] = in_data[(b * N + i) * 2];
                buf[2 * i + 1] = in_data[(b * N + i) * 2 + 1];
            }

            fft_detail::cooley_tukey(buf.data(), padded_n, false);

            // Copy to output
            for (int64_t i = 0; i < padded_n; ++i) {
                out_data[(b * padded_n + i) * 2] = static_cast<float>(buf[2 * i]);
                out_data[(b * padded_n + i) * 2 + 1] = static_cast<float>(buf[2 * i + 1]);
            }
        }

        return output;
    }

    // General case: move fft_dim to position ndim-2 (before complex dim)
    // Then process as above
    {
        std::vector<int64_t> perm;
        for (int64_t d = 0; d < ndim; ++d) {
            if (d != fft_dim && d != ndim - 1) perm.push_back(d);
        }
        perm.push_back(fft_dim);
        perm.push_back(ndim - 1);

        Tensor permuted = at::native::permute(complex_input, perm).contiguous();
        int64_t batch = permuted.numel() / (N * 2);

        auto pout_shape = permuted.sizes().vec();
        pout_shape[pout_shape.size() - 2] = padded_n;
        Tensor pout = at::zeros(pout_shape);

        const float* p_in = permuted.data_ptr<float>();
        float* p_out = pout.mutable_data_ptr<float>();

        for (int64_t b = 0; b < batch; ++b) {
            std::vector<double> buf(2 * padded_n, 0.0);
            for (int64_t i = 0; i < std::min(N, target_n); ++i) {
                buf[2 * i] = p_in[(b * N + i) * 2];
                buf[2 * i + 1] = p_in[(b * N + i) * 2 + 1];
            }
            fft_detail::cooley_tukey(buf.data(), padded_n, false);
            for (int64_t i = 0; i < padded_n; ++i) {
                p_out[(b * padded_n + i) * 2] = static_cast<float>(buf[2 * i]);
                p_out[(b * padded_n + i) * 2 + 1] = static_cast<float>(buf[2 * i + 1]);
            }
        }

        // Inverse permute
        std::vector<int64_t> inv_perm(ndim);
        for (int64_t i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
        return at::native::permute(pout, inv_perm).contiguous();
    }
}

// ============================================================================
// ifft: inverse FFT
// ============================================================================

inline Tensor ifft(const Tensor& self, int64_t n = -1, int64_t dim = -1) {
    Tensor input = self.contiguous();
    int64_t ndim = input.dim();
    PT_CHECK_MSG(ndim >= 2 && input.size(-1) == 2, "ifft requires complex input [..., 2]");

    if (dim == -1) {
        dim = ndim - 2;  // default: last data dim before complex pair
    } else if (dim < 0) {
        dim += ndim;
    }

    int64_t fft_dim = dim;
    int64_t N = input.size(fft_dim);
    int64_t target_n = (n > 0) ? n : N;
    int64_t padded_n = fft_detail::next_power_of_2(target_n);

    // Move fft_dim to -2
    std::vector<int64_t> perm;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != fft_dim && d != ndim - 1) perm.push_back(d);
    }
    perm.push_back(fft_dim);
    perm.push_back(ndim - 1);

    Tensor permuted = at::native::permute(input, perm).contiguous();
    int64_t batch = permuted.numel() / (N * 2);

    auto pout_shape = permuted.sizes().vec();
    pout_shape[pout_shape.size() - 2] = padded_n;
    Tensor pout = at::zeros(pout_shape);

    const float* p_in = permuted.data_ptr<float>();
    float* p_out = pout.mutable_data_ptr<float>();

    for (int64_t b = 0; b < batch; ++b) {
        std::vector<double> buf(2 * padded_n, 0.0);
        for (int64_t i = 0; i < std::min(N, target_n); ++i) {
            buf[2 * i] = p_in[(b * N + i) * 2];
            buf[2 * i + 1] = p_in[(b * N + i) * 2 + 1];
        }
        fft_detail::cooley_tukey(buf.data(), padded_n, true);  // inverse
        for (int64_t i = 0; i < padded_n; ++i) {
            p_out[(b * padded_n + i) * 2] = static_cast<float>(buf[2 * i]);
            p_out[(b * padded_n + i) * 2 + 1] = static_cast<float>(buf[2 * i + 1]);
        }
    }

    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
    return at::native::permute(pout, inv_perm).contiguous();
}

// ============================================================================
// rfft: real-to-complex FFT
// Input: [..., N] (real), Output: [..., N/2+1, 2] (complex)
// ============================================================================

inline Tensor rfft(const Tensor& self, int64_t n = -1, int64_t dim = -1) {
    Tensor input = self.contiguous();
    int64_t ndim = input.dim();
    if (dim < 0) dim += ndim;

    int64_t N = input.size(dim);
    int64_t target_n = (n > 0) ? n : N;
    int64_t padded_n = fft_detail::next_power_of_2(target_n);
    int64_t out_n = padded_n / 2 + 1;

    // Add complex dim and do full FFT
    Tensor zeros_t = at::zeros(input.sizes());
    Tensor complex_input = at::native::stack({input, zeros_t}, -1);

    Tensor full_result = fft(complex_input, target_n, dim);

    // Take only first N/2+1 elements along fft_dim
    return full_result.narrow(dim, 0, out_n);
}

// ============================================================================
// irfft: complex-to-real inverse FFT
// Input: [..., N/2+1, 2] (complex), Output: [..., N] (real)
// ============================================================================

inline Tensor irfft(const Tensor& self, int64_t n = -1, int64_t dim = -1) {
    Tensor input = self.contiguous();
    int64_t ndim = input.dim();
    if (dim < 0) dim += ndim;

    PT_CHECK_MSG(input.size(-1) == 2, "irfft requires complex input [..., 2]");

    int64_t half_n = input.size(dim);
    int64_t target_n = (n > 0) ? n : 2 * (half_n - 1);
    int64_t padded_n = fft_detail::next_power_of_2(target_n);

    // Reconstruct full spectrum using conjugate symmetry
    // Move dim to -2, process, move back
    std::vector<int64_t> perm;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim && d != ndim - 1) perm.push_back(d);
    }
    perm.push_back(dim);
    perm.push_back(ndim - 1);

    Tensor permuted = at::native::permute(input, perm).contiguous();
    int64_t batch = permuted.numel() / (half_n * 2);

    auto pout_shape = permuted.sizes().vec();
    pout_shape[pout_shape.size() - 2] = padded_n;
    Tensor full_perm = at::zeros(pout_shape);

    const float* p_in = permuted.data_ptr<float>();
    float* p_out = full_perm.mutable_data_ptr<float>();

    for (int64_t b = 0; b < batch; ++b) {
        // Copy first half
        for (int64_t i = 0; i < half_n; ++i) {
            p_out[(b * padded_n + i) * 2] = p_in[(b * half_n + i) * 2];
            p_out[(b * padded_n + i) * 2 + 1] = p_in[(b * half_n + i) * 2 + 1];
        }
        // Conjugate symmetry for second half
        for (int64_t i = 1; i < padded_n - half_n + 1; ++i) {
            int64_t mirror = padded_n - i;
            if (mirror < padded_n && i < half_n) {
                p_out[(b * padded_n + mirror) * 2] = p_in[(b * half_n + i) * 2];
                p_out[(b * padded_n + mirror) * 2 + 1] = -p_in[(b * half_n + i) * 2 + 1];
            }
        }
    }

    // Do inverse FFT
    for (int64_t b = 0; b < batch; ++b) {
        std::vector<double> buf(2 * padded_n);
        for (int64_t i = 0; i < padded_n; ++i) {
            buf[2 * i] = p_out[(b * padded_n + i) * 2];
            buf[2 * i + 1] = p_out[(b * padded_n + i) * 2 + 1];
        }
        fft_detail::cooley_tukey(buf.data(), padded_n, true);
        for (int64_t i = 0; i < padded_n; ++i) {
            p_out[(b * padded_n + i) * 2] = static_cast<float>(buf[2 * i]);
            p_out[(b * padded_n + i) * 2 + 1] = static_cast<float>(buf[2 * i + 1]);
        }
    }

    // Extract real part
    std::vector<int64_t> inv_perm(ndim);
    for (int64_t i = 0; i < ndim; ++i) inv_perm[perm[i]] = i;
    Tensor result_complex = at::native::permute(full_perm, inv_perm).contiguous();

    // Select real part (index 0 along last dim)
    return result_complex.select(-1, 0).narrow(dim, 0, target_n).contiguous();
}

// ============================================================================
// fft2 / ifft2: 2D FFT
// ============================================================================

inline Tensor fft2(const Tensor& self) {
    // Apply FFT along last two spatial dims
    // For [..., H, W, 2]: fft along dim -3, then dim -2
    Tensor result = fft(self, -1, -3);
    result = fft(result, -1, -2);
    return result;
}

inline Tensor ifft2(const Tensor& self) {
    Tensor result = ifft(self, -1, -3);
    result = ifft(result, -1, -2);
    return result;
}

// ============================================================================
// fftfreq / rfftfreq: frequency bins
// ============================================================================

inline Tensor fftfreq(int64_t n, double d = 1.0, const TensorOptions& options = TensorOptions()) {
    Tensor result = empty({n}, options);
    float* data = result.mutable_data_ptr<float>();

    double val = 1.0 / (n * d);
    for (int64_t i = 0; i < (n + 1) / 2; ++i) {
        data[i] = static_cast<float>(i * val);
    }
    for (int64_t i = (n + 1) / 2; i < n; ++i) {
        data[i] = static_cast<float>((i - n) * val);
    }

    return result;
}

inline Tensor rfftfreq(int64_t n, double d = 1.0, const TensorOptions& options = TensorOptions()) {
    int64_t out_n = n / 2 + 1;
    Tensor result = empty({out_n}, options);
    float* data = result.mutable_data_ptr<float>();

    double val = 1.0 / (n * d);
    for (int64_t i = 0; i < out_n; ++i) {
        data[i] = static_cast<float>(i * val);
    }

    return result;
}

// ============================================================================
// fftshift / ifftshift: shift zero-frequency to center
// ============================================================================

inline Tensor fftshift(const Tensor& self, c10::IntArrayRef dims = {}) {
    std::vector<int64_t> shift_dims;
    if (dims.empty()) {
        // All dims except last (complex dim)
        for (int64_t d = 0; d < self.dim() - 1; ++d) {
            shift_dims.push_back(d);
        }
    } else {
        shift_dims = dims.vec();
    }

    std::vector<int64_t> shifts;
    for (auto d : shift_dims) {
        int64_t actual_dim = d < 0 ? d + self.dim() : d;
        shifts.push_back(self.size(actual_dim) / 2);
    }

    return roll(self, shifts, shift_dims);
}

inline Tensor ifftshift(const Tensor& self, c10::IntArrayRef dims = {}) {
    std::vector<int64_t> shift_dims;
    if (dims.empty()) {
        for (int64_t d = 0; d < self.dim() - 1; ++d) {
            shift_dims.push_back(d);
        }
    } else {
        shift_dims = dims.vec();
    }

    std::vector<int64_t> shifts;
    for (auto d : shift_dims) {
        int64_t actual_dim = d < 0 ? d + self.dim() : d;
        shifts.push_back(-(self.size(actual_dim) / 2));
    }

    return roll(self, shifts, shift_dims);
}

} // namespace native
} // namespace at
