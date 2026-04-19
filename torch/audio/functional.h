// torch/audio/functional.h
// torchaudio.functional compatibility shim for PromeTorch.
//
// CPU-only, header-only. Safe on Elbrus (LCC) — no intrinsics, stdlib only.
// Tensor dtype: float32 throughout.
//
// Provides:
//   stft, istft, spectrogram, melscale_fbanks, resample,
//   mu_law_encoding, mu_law_decoding.
//
// STFT output layout: [..., n_freq, n_frames, 2] where last dim = (real, imag).
// n_freq = n_fft / 2 + 1 (one-sided).
//
// FFT is a Cooley-Tukey radix-2 iterative implementation. If n_fft is not
// a power of two we fall back to a naive O(N^2) DFT.
#pragma once

#include "aten/src/ATen/ATen.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace torch {
namespace audio {
namespace functional {

using at::Tensor;

namespace detail {

constexpr double kPI = 3.14159265358979323846;

inline bool is_pow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

inline int ilog2(int n) {
    int r = 0;
    while ((1 << r) < n) ++r;
    return r;
}

// In-place iterative radix-2 FFT. Input: complex array of length N (pow2),
// interleaved as re, im, re, im, ...
// inverse=true performs the inverse transform without 1/N normalization.
inline void fft_radix2_inplace(float* data, int N, bool inverse) {
    const int logN = ilog2(N);
    // Bit reversal
    for (int i = 1, j = 0; i < N; ++i) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(data[2 * i + 0], data[2 * j + 0]);
            std::swap(data[2 * i + 1], data[2 * j + 1]);
        }
    }
    (void)logN;
    const double sign = inverse ? +1.0 : -1.0;
    for (int len = 2; len <= N; len <<= 1) {
        const double ang = sign * 2.0 * kPI / static_cast<double>(len);
        const double wlen_re = std::cos(ang);
        const double wlen_im = std::sin(ang);
        const int half = len >> 1;
        for (int i = 0; i < N; i += len) {
            double w_re = 1.0, w_im = 0.0;
            for (int k = 0; k < half; ++k) {
                const int a = 2 * (i + k);
                const int b = 2 * (i + k + half);
                const double u_re = data[a];
                const double u_im = data[a + 1];
                const double v_re = data[b] * w_re - data[b + 1] * w_im;
                const double v_im = data[b] * w_im + data[b + 1] * w_re;
                data[a] = static_cast<float>(u_re + v_re);
                data[a + 1] = static_cast<float>(u_im + v_im);
                data[b] = static_cast<float>(u_re - v_re);
                data[b + 1] = static_cast<float>(u_im - v_im);
                const double nw_re = w_re * wlen_re - w_im * wlen_im;
                const double nw_im = w_re * wlen_im + w_im * wlen_re;
                w_re = nw_re;
                w_im = nw_im;
            }
        }
    }
}

// Naive O(N^2) DFT fallback for non-power-of-2 n_fft.
inline void dft_naive(const float* in, float* out, int N, bool inverse) {
    const double sign = inverse ? +1.0 : -1.0;
    for (int k = 0; k < N; ++k) {
        double acc_re = 0.0, acc_im = 0.0;
        for (int n = 0; n < N; ++n) {
            const double ang = sign * 2.0 * kPI * k * n / static_cast<double>(N);
            const double c = std::cos(ang), s = std::sin(ang);
            const double xr = in[2 * n + 0];
            const double xi = in[2 * n + 1];
            acc_re += xr * c - xi * s;
            acc_im += xr * s + xi * c;
        }
        out[2 * k + 0] = static_cast<float>(acc_re);
        out[2 * k + 1] = static_cast<float>(acc_im);
    }
}

// 1D FFT forward on a real-valued frame. Output has length n_fft * 2 (complex).
inline void fft1d(const float* frame, int n_fft, float* out_cplx, bool inverse) {
    if (is_pow2(n_fft)) {
        for (int i = 0; i < n_fft; ++i) {
            out_cplx[2 * i + 0] = frame[2 * i + 0];
            out_cplx[2 * i + 1] = frame[2 * i + 1];
        }
        fft_radix2_inplace(out_cplx, n_fft, inverse);
    } else {
        dft_naive(frame, out_cplx, n_fft, inverse);
    }
}

// Hann window of length L (periodic form, matches torchaudio default).
inline Tensor hann_window(int64_t L) {
    Tensor w = at::empty({L}, at::TensorOptions().dtype(c10::ScalarType::Float));
    float* p = w.mutable_data_ptr<float>();
    for (int64_t i = 0; i < L; ++i) {
        p[i] = static_cast<float>(0.5 * (1.0 - std::cos(2.0 * kPI * i / static_cast<double>(L))));
    }
    return w;
}

} // namespace detail

// ---------------------------------------------------------------------------
// STFT
// ---------------------------------------------------------------------------
// Input:  1-D waveform [samples] or 2-D [channels, samples].
// Output: [n_freq, n_frames, 2] (1-D input) or [channels, n_freq, n_frames, 2].
// window: optional 1-D tensor of length win_length; if empty -> Hann window.
// center=true pads the signal with reflection so frames are centered on t*hop.
inline Tensor stft(
    const Tensor& input,
    int64_t n_fft,
    int64_t hop_length = 0,
    int64_t win_length = 0,
    const Tensor& window = Tensor(),
    bool center = true
) {
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    Tensor win = window.defined() ? window : detail::hann_window(win_length);
    const float* win_p = win.data_ptr<float>();

    // Flatten input to [channels, samples].
    Tensor x = input.contiguous();
    int64_t channels = 1, samples = 0;
    bool batched = (x.dim() == 2);
    if (batched) { channels = x.size(0); samples = x.size(1); }
    else         { samples  = x.size(0); }

    // Reflection pad if center=true by n_fft/2 on each side.
    const int64_t pad = center ? (n_fft / 2) : 0;
    const int64_t padded_len = samples + 2 * pad;

    std::vector<float> padded(static_cast<size_t>(channels * padded_len), 0.0f);
    const float* src_p = x.data_ptr<float>();
    for (int64_t c = 0; c < channels; ++c) {
        const float* src = src_p + c * samples;
        float* dst = padded.data() + c * padded_len;
        for (int64_t i = 0; i < pad; ++i) {
            const int64_t j = pad - i;
            dst[i] = (j < samples) ? src[j] : 0.0f;
        }
        for (int64_t i = 0; i < samples; ++i) {
            dst[pad + i] = src[i];
        }
        for (int64_t i = 0; i < pad; ++i) {
            const int64_t j = samples - 2 - i;
            dst[pad + samples + i] = (j >= 0 && j < samples) ? src[j] : 0.0f;
        }
    }

    const int64_t n_frames = (padded_len >= n_fft)
        ? ((padded_len - n_fft) / hop_length + 1)
        : 0;
    const int64_t n_freq = n_fft / 2 + 1;

    std::vector<int64_t> out_shape;
    if (batched) out_shape = {channels, n_freq, n_frames, 2};
    else         out_shape = {n_freq, n_frames, 2};

    Tensor out = at::zeros(out_shape, at::TensorOptions().dtype(c10::ScalarType::Float));
    float* out_p = out.mutable_data_ptr<float>();

    std::vector<float> frame(static_cast<size_t>(2 * n_fft), 0.0f);
    std::vector<float> spec(static_cast<size_t>(2 * n_fft), 0.0f);

    // Window is centered inside the FFT window if win_length < n_fft.
    const int64_t win_offset = (n_fft - win_length) / 2;

    for (int64_t c = 0; c < channels; ++c) {
        const float* sig = padded.data() + c * padded_len;
        float* out_c = out_p + (batched ? c * n_freq * n_frames * 2 : 0);
        for (int64_t f = 0; f < n_frames; ++f) {
            std::fill(frame.begin(), frame.end(), 0.0f);
            const int64_t start = f * hop_length;
            for (int64_t i = 0; i < win_length; ++i) {
                const int64_t idx = start + win_offset + i;
                if (idx < padded_len) {
                    frame[2 * (win_offset + i)] = sig[idx] * win_p[i];
                }
            }
            detail::fft1d(frame.data(), static_cast<int>(n_fft), spec.data(), /*inverse=*/false);
            for (int64_t k = 0; k < n_freq; ++k) {
                const int64_t idx = (k * n_frames + f) * 2;
                out_c[idx + 0] = spec[2 * k + 0];
                out_c[idx + 1] = spec[2 * k + 1];
            }
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// iSTFT
// ---------------------------------------------------------------------------
// Input shape: [n_freq, n_frames, 2] or [channels, n_freq, n_frames, 2].
// length: desired output length (post trim). If <=0, auto.
inline Tensor istft(
    const Tensor& stft_tensor,
    int64_t n_fft,
    int64_t hop_length = 0,
    int64_t win_length = 0,
    const Tensor& window = Tensor(),
    bool center = true,
    int64_t length = -1
) {
    if (hop_length <= 0) hop_length = n_fft / 4;
    if (win_length <= 0) win_length = n_fft;

    Tensor win = window.defined() ? window : detail::hann_window(win_length);
    const float* win_p = win.data_ptr<float>();

    Tensor s = stft_tensor.contiguous();
    bool batched = (s.dim() == 4);
    int64_t channels = 1, n_freq = 0, n_frames = 0;
    if (batched) { channels = s.size(0); n_freq = s.size(1); n_frames = s.size(2); }
    else         { n_freq   = s.size(0); n_frames = s.size(1); }

    const int64_t pad = center ? (n_fft / 2) : 0;
    const int64_t padded_len = (n_frames - 1) * hop_length + n_fft;

    std::vector<float> out_buf(static_cast<size_t>(channels * padded_len), 0.0f);
    std::vector<float> wsum_buf(static_cast<size_t>(channels * padded_len), 0.0f);

    const float* s_p = s.data_ptr<float>();

    std::vector<float> full(static_cast<size_t>(2 * n_fft), 0.0f);
    std::vector<float> time(static_cast<size_t>(2 * n_fft), 0.0f);

    const int64_t win_offset = (n_fft - win_length) / 2;

    // Precompute squared window for overlap normalization.
    std::vector<float> win_sq(static_cast<size_t>(win_length));
    for (int64_t i = 0; i < win_length; ++i) win_sq[i] = win_p[i] * win_p[i];

    for (int64_t c = 0; c < channels; ++c) {
        const float* s_c = s_p + (batched ? c * n_freq * n_frames * 2 : 0);
        float* out_c  = out_buf.data()  + c * padded_len;
        float* wsum_c = wsum_buf.data() + c * padded_len;
        for (int64_t f = 0; f < n_frames; ++f) {
            // Build full spectrum from one-sided (Hermitian symmetry).
            std::fill(full.begin(), full.end(), 0.0f);
            for (int64_t k = 0; k < n_freq; ++k) {
                const int64_t idx = (k * n_frames + f) * 2;
                full[2 * k + 0] = s_c[idx + 0];
                full[2 * k + 1] = s_c[idx + 1];
            }
            for (int64_t k = 1; k < n_fft - n_freq + 1; ++k) {
                const int64_t src = n_freq - 1 - k;
                if (src < 0 || src >= n_freq) continue;
                const int64_t dst = n_freq - 1 + k;
                if (dst < 0 || dst >= n_fft) continue;
                const int64_t sidx = (src * n_frames + f) * 2;
                full[2 * dst + 0] =  s_c[sidx + 0];
                full[2 * dst + 1] = -s_c[sidx + 1];
            }
            detail::fft1d(full.data(), static_cast<int>(n_fft), time.data(), /*inverse=*/true);
            const float inv_n = 1.0f / static_cast<float>(n_fft);
            const int64_t base = f * hop_length;
            for (int64_t i = 0; i < win_length; ++i) {
                const int64_t dst_idx = base + win_offset + i;
                if (dst_idx >= padded_len) break;
                const float sample = time[2 * (win_offset + i)] * inv_n;
                out_c[dst_idx]  += sample * win_p[i];
                wsum_c[dst_idx] += win_sq[i];
            }
        }
        for (int64_t i = 0; i < padded_len; ++i) {
            if (wsum_c[i] > 1e-11f) out_c[i] /= wsum_c[i];
        }
    }

    // Trim center padding.
    int64_t out_len = padded_len - 2 * pad;
    if (length > 0) out_len = length;

    std::vector<int64_t> out_shape = batched
        ? std::vector<int64_t>{channels, out_len}
        : std::vector<int64_t>{out_len};
    Tensor out = at::empty(out_shape, at::TensorOptions().dtype(c10::ScalarType::Float));
    float* op = out.mutable_data_ptr<float>();
    for (int64_t c = 0; c < channels; ++c) {
        const float* src = out_buf.data() + c * padded_len + pad;
        float* dst = op + c * out_len;
        const int64_t copy = std::min(out_len, padded_len - pad);
        for (int64_t i = 0; i < copy; ++i) dst[i] = src[i];
        for (int64_t i = copy; i < out_len; ++i) dst[i] = 0.0f;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Spectrogram
// ---------------------------------------------------------------------------
// power=1 -> magnitude, power=2 -> power, power<=0 -> complex (same as stft).
inline Tensor spectrogram(
    const Tensor& input,
    int64_t n_fft,
    int64_t win_length = 0,
    int64_t hop_length = 0,
    double power = 2.0,
    const Tensor& window = Tensor(),
    bool center = true
) {
    Tensor s = stft(input, n_fft, hop_length, win_length, window, center);
    if (power <= 0.0) return s;

    // Reduce last dim (2) -> magnitude/power.
    std::vector<int64_t> shape = s.sizes().vec();
    shape.pop_back(); // drop 2
    Tensor out = at::empty(shape, at::TensorOptions().dtype(c10::ScalarType::Float));
    const float* src = s.data_ptr<float>();
    float* dst = out.mutable_data_ptr<float>();
    const int64_t n = out.numel();
    if (power == 2.0) {
        for (int64_t i = 0; i < n; ++i) {
            const float re = src[2 * i + 0];
            const float im = src[2 * i + 1];
            dst[i] = re * re + im * im;
        }
    } else if (power == 1.0) {
        for (int64_t i = 0; i < n; ++i) {
            const float re = src[2 * i + 0];
            const float im = src[2 * i + 1];
            dst[i] = std::sqrt(re * re + im * im);
        }
    } else {
        for (int64_t i = 0; i < n; ++i) {
            const float re = src[2 * i + 0];
            const float im = src[2 * i + 1];
            dst[i] = std::pow(std::sqrt(re * re + im * im), static_cast<float>(power));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Mel filterbanks
// ---------------------------------------------------------------------------
// Returns [n_stft, n_mels] so that mel_spec = spec @ fbanks.
// n_stft = n_fft/2 + 1.  HTK mel scale (torchaudio default = 'htk').
inline Tensor melscale_fbanks(
    int64_t n_mels,
    double f_min,
    double f_max,
    int64_t n_stft,
    double sample_rate
) {
    auto hz_to_mel = [](double hz) {
        return 2595.0 * std::log10(1.0 + hz / 700.0);
    };
    auto mel_to_hz = [](double mel) {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    };

    const double m_min = hz_to_mel(f_min);
    const double m_max = hz_to_mel(f_max);

    std::vector<double> mel_pts(static_cast<size_t>(n_mels + 2));
    for (int64_t i = 0; i < n_mels + 2; ++i) {
        mel_pts[i] = m_min + (m_max - m_min) * i / (n_mels + 1);
    }
    std::vector<double> f_pts(mel_pts.size());
    for (size_t i = 0; i < mel_pts.size(); ++i) f_pts[i] = mel_to_hz(mel_pts[i]);

    // Bin center frequencies of the STFT.
    std::vector<double> bin_hz(static_cast<size_t>(n_stft));
    for (int64_t i = 0; i < n_stft; ++i) {
        bin_hz[i] = sample_rate * i / (2.0 * (n_stft - 1));
    }

    Tensor fb = at::zeros({n_stft, n_mels}, at::TensorOptions().dtype(c10::ScalarType::Float));
    float* p = fb.mutable_data_ptr<float>();

    for (int64_t m = 0; m < n_mels; ++m) {
        const double lo = f_pts[m];
        const double ce = f_pts[m + 1];
        const double hi = f_pts[m + 2];
        for (int64_t k = 0; k < n_stft; ++k) {
            const double f = bin_hz[k];
            double w = 0.0;
            if (f >= lo && f <= ce && ce > lo) w = (f - lo) / (ce - lo);
            else if (f >= ce && f <= hi && hi > ce) w = (hi - f) / (hi - ce);
            if (w > 0.0) p[k * n_mels + m] = static_cast<float>(w);
        }
    }
    return fb;
}

// ---------------------------------------------------------------------------
// Resample (polyphase rational, windowed sinc)
// ---------------------------------------------------------------------------
// Simple high-quality resampler: builds an ideal lowpass sinc kernel bandlimited
// to min(orig_freq, new_freq)/2 and resamples by rational L/M factors reduced to gcd.
inline Tensor resample(
    const Tensor& input,
    int64_t orig_freq,
    int64_t new_freq,
    int64_t lowpass_filter_width = 6
) {
    auto gcd = [](int64_t a, int64_t b) {
        while (b) { int64_t t = a % b; a = b; b = t; }
        return a;
    };
    const int64_t g = gcd(orig_freq, new_freq);
    const int64_t L = new_freq / g;  // upsample
    const int64_t M = orig_freq / g; // downsample

    Tensor x = input.contiguous();
    bool batched = (x.dim() == 2);
    int64_t channels = batched ? x.size(0) : 1;
    int64_t n = batched ? x.size(1) : x.size(0);
    const float* xp = x.data_ptr<float>();

    const double cutoff = 0.5 / static_cast<double>(std::max<int64_t>(L, M));
    const int64_t half_len = lowpass_filter_width * std::max<int64_t>(L, M);

    const int64_t out_n = (n * L + M - 1) / M;
    std::vector<int64_t> out_shape = batched
        ? std::vector<int64_t>{channels, out_n}
        : std::vector<int64_t>{out_n};
    Tensor out = at::zeros(out_shape, at::TensorOptions().dtype(c10::ScalarType::Float));
    float* op = out.mutable_data_ptr<float>();

    for (int64_t c = 0; c < channels; ++c) {
        const float* in_c = xp + c * n;
        float* out_c = op + c * out_n;
        for (int64_t i = 0; i < out_n; ++i) {
            // Target sample position in input domain (fractional).
            const double t_in = static_cast<double>(i) * M / static_cast<double>(L);
            const int64_t t0 = static_cast<int64_t>(std::floor(t_in));
            double acc = 0.0, norm = 0.0;
            for (int64_t k = -half_len; k <= half_len; ++k) {
                const int64_t idx = t0 + k;
                if (idx < 0 || idx >= n) continue;
                const double dt = t_in - static_cast<double>(idx);
                // Hann-windowed sinc
                const double window_val = 0.5 * (1.0 + std::cos(detail::kPI * dt / static_cast<double>(half_len)));
                if (std::abs(dt) >= static_cast<double>(half_len)) continue;
                double sinc;
                if (std::abs(dt) < 1e-9) sinc = 2.0 * cutoff;
                else sinc = std::sin(2.0 * detail::kPI * cutoff * dt) / (detail::kPI * dt);
                const double w = sinc * window_val;
                acc  += static_cast<double>(in_c[idx]) * w;
                norm += w;
            }
            out_c[i] = static_cast<float>(norm > 1e-12 ? acc / norm : 0.0);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// μ-law codec
// ---------------------------------------------------------------------------
// Input: float in [-1, 1]. Output: int64 in [0, quantization_channels-1].
inline Tensor mu_law_encoding(const Tensor& input, int64_t quantization_channels = 256) {
    const double mu = static_cast<double>(quantization_channels - 1);
    Tensor x = input.contiguous();
    Tensor out = at::empty(x.sizes(), at::TensorOptions().dtype(c10::ScalarType::Long));
    const float* src = x.data_ptr<float>();
    int64_t* dst = out.mutable_data_ptr<int64_t>();
    const int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i) {
        double v = std::max(-1.0, std::min(1.0, static_cast<double>(src[i])));
        const double s = (v < 0) ? -1.0 : 1.0;
        const double y = s * std::log(1.0 + mu * std::abs(v)) / std::log(1.0 + mu);
        int64_t q = static_cast<int64_t>(std::floor((y + 1.0) / 2.0 * mu + 0.5));
        if (q < 0) q = 0;
        if (q > quantization_channels - 1) q = quantization_channels - 1;
        dst[i] = q;
    }
    return out;
}

inline Tensor mu_law_decoding(const Tensor& input, int64_t quantization_channels = 256) {
    const double mu = static_cast<double>(quantization_channels - 1);
    Tensor x = input.contiguous();
    Tensor out = at::empty(x.sizes(), at::TensorOptions().dtype(c10::ScalarType::Float));
    float* dst = out.mutable_data_ptr<float>();
    const int64_t n = x.numel();
    // Accept either int64 or float input.
    if (x.dtype() == c10::ScalarType::Long) {
        const int64_t* src = x.data_ptr<int64_t>();
        for (int64_t i = 0; i < n; ++i) {
            const double y = 2.0 * src[i] / mu - 1.0;
            const double s = (y < 0) ? -1.0 : 1.0;
            dst[i] = static_cast<float>(s * (std::pow(1.0 + mu, std::abs(y)) - 1.0) / mu);
        }
    } else {
        const float* src = x.data_ptr<float>();
        for (int64_t i = 0; i < n; ++i) {
            const double y = 2.0 * src[i] / mu - 1.0;
            const double s = (y < 0) ? -1.0 : 1.0;
            dst[i] = static_cast<float>(s * (std::pow(1.0 + mu, std::abs(y)) - 1.0) / mu);
        }
    }
    return out;
}

} // namespace functional
} // namespace audio
} // namespace torch
