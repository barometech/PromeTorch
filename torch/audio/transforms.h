// torch/audio/transforms.h
// torchaudio.transforms compatibility shim for PromeTorch.
//
// Module wrappers around torch::audio::functional. Header-only, CPU-only.
#pragma once

#include "torch/audio/functional.h"
#include "torch/nn/module.h"
#include "aten/src/ATen/ATen.h"

#include <cmath>
#include <memory>
#include <vector>

namespace torch {
namespace audio {
namespace transforms {

using at::Tensor;
using torch::nn::Module;

// ---------------------------------------------------------------------------
// Spectrogram
// ---------------------------------------------------------------------------
class Spectrogram : public Module {
public:
    Spectrogram(
        int64_t n_fft = 400,
        int64_t win_length = 0,
        int64_t hop_length = 0,
        double  power = 2.0,
        bool    center = true
    )
        : Module("Spectrogram")
        , n_fft_(n_fft)
        , win_length_(win_length > 0 ? win_length : n_fft)
        , hop_length_(hop_length > 0 ? hop_length : n_fft / 4)
        , power_(power)
        , center_(center)
        , window_(functional::detail::hann_window(win_length > 0 ? win_length : n_fft))
    {}

    Tensor forward(const Tensor& waveform) override {
        return functional::spectrogram(waveform, n_fft_, win_length_, hop_length_,
                                       power_, window_, center_);
    }

    int64_t n_fft() const { return n_fft_; }

private:
    int64_t n_fft_, win_length_, hop_length_;
    double  power_;
    bool    center_;
    Tensor  window_;
};

// ---------------------------------------------------------------------------
// MelSpectrogram
// ---------------------------------------------------------------------------
class MelSpectrogram : public Module {
public:
    MelSpectrogram(
        int64_t sample_rate = 16000,
        int64_t n_fft = 400,
        int64_t win_length = 0,
        int64_t hop_length = 0,
        double  f_min = 0.0,
        double  f_max = -1.0,
        int64_t n_mels = 128,
        bool    center = true
    )
        : Module("MelSpectrogram")
        , sample_rate_(sample_rate)
        , n_fft_(n_fft)
        , win_length_(win_length > 0 ? win_length : n_fft)
        , hop_length_(hop_length > 0 ? hop_length : n_fft / 4)
        , f_min_(f_min)
        , f_max_(f_max > 0 ? f_max : sample_rate / 2.0)
        , n_mels_(n_mels)
        , center_(center)
        , window_(functional::detail::hann_window(win_length > 0 ? win_length : n_fft))
        , fbanks_(functional::melscale_fbanks(n_mels, f_min_, f_max_,
                                              n_fft / 2 + 1, static_cast<double>(sample_rate)))
    {}

    // Input: [..., samples] -> Output: [..., n_mels, n_frames]
    Tensor forward(const Tensor& waveform) override {
        Tensor spec = functional::spectrogram(
            waveform, n_fft_, win_length_, hop_length_, 2.0, window_, center_);

        // spec shape: [..., n_freq, n_frames]; fbanks: [n_freq, n_mels].
        // mel = fbanks^T @ spec  -> [..., n_mels, n_frames]
        const bool batched = (spec.dim() == 3);
        Tensor spec2 = spec.contiguous();
        const int64_t n_freq   = spec2.size(batched ? 1 : 0);
        const int64_t n_frames = spec2.size(batched ? 2 : 1);
        const int64_t batch    = batched ? spec2.size(0) : 1;

        std::vector<int64_t> out_shape = batched
            ? std::vector<int64_t>{batch, n_mels_, n_frames}
            : std::vector<int64_t>{n_mels_, n_frames};
        Tensor out = at::zeros(out_shape, at::TensorOptions().dtype(c10::ScalarType::Float));
        float* op = out.mutable_data_ptr<float>();
        const float* sp = spec2.data_ptr<float>();
        const float* fb = fbanks_.data_ptr<float>();

        for (int64_t b = 0; b < batch; ++b) {
            const float* sp_b = sp + b * n_freq * n_frames;
            float* op_b = op + b * n_mels_ * n_frames;
            for (int64_t m = 0; m < n_mels_; ++m) {
                for (int64_t t = 0; t < n_frames; ++t) {
                    double acc = 0.0;
                    for (int64_t k = 0; k < n_freq; ++k) {
                        acc += fb[k * n_mels_ + m] * sp_b[k * n_frames + t];
                    }
                    op_b[m * n_frames + t] = static_cast<float>(acc);
                }
            }
        }
        return out;
    }

    const Tensor& mel_filterbanks() const { return fbanks_; }

private:
    int64_t sample_rate_, n_fft_, win_length_, hop_length_;
    double  f_min_, f_max_;
    int64_t n_mels_;
    bool    center_;
    Tensor  window_, fbanks_;
};

// ---------------------------------------------------------------------------
// AmplitudeToDB
// ---------------------------------------------------------------------------
// stype='amplitude' -> multiplier=20, 'power' -> multiplier=10.
class AmplitudeToDB : public Module {
public:
    AmplitudeToDB(const std::string& stype = "power", double top_db = -1.0)
        : Module("AmplitudeToDB")
        , multiplier_(stype == "amplitude" ? 20.0 : 10.0)
        , amin_(1e-10)
        , db_multiplier_(0.0)
        , top_db_(top_db)
    {}

    Tensor forward(const Tensor& x) override {
        Tensor xc = x.contiguous();
        Tensor out = at::empty(xc.sizes(), at::TensorOptions().dtype(c10::ScalarType::Float));
        const float* src = xc.data_ptr<float>();
        float* dst = out.mutable_data_ptr<float>();
        const int64_t n = xc.numel();
        float maxv = -1e30f;
        for (int64_t i = 0; i < n; ++i) {
            const double v = std::max<double>(amin_, static_cast<double>(src[i]));
            dst[i] = static_cast<float>(multiplier_ * std::log10(v));
            if (dst[i] > maxv) maxv = dst[i];
        }
        if (top_db_ > 0.0) {
            const float floor_v = maxv - static_cast<float>(top_db_);
            for (int64_t i = 0; i < n; ++i) {
                if (dst[i] < floor_v) dst[i] = floor_v;
            }
        }
        return out;
    }

private:
    double multiplier_, amin_, db_multiplier_, top_db_;
};

// ---------------------------------------------------------------------------
// MFCC — MelSpec (power) -> log -> DCT-II along mel dim.
// ---------------------------------------------------------------------------
class MFCC : public Module {
public:
    MFCC(
        int64_t sample_rate = 16000,
        int64_t n_mfcc = 40,
        int64_t n_fft = 400,
        int64_t hop_length = 0,
        int64_t n_mels = 128
    )
        : Module("MFCC")
        , n_mfcc_(n_mfcc)
        , n_mels_(n_mels)
        , mel_(sample_rate, n_fft, 0, hop_length, 0.0, -1.0, n_mels, true)
        , to_db_("power", 80.0)
    {
        build_dct_matrix();
    }

    Tensor forward(const Tensor& waveform) override {
        Tensor mel = mel_.forward(waveform);            // [..., n_mels, T]
        Tensor db  = to_db_.forward(mel);               // log-scaled
        // mfcc = dct_mat @ db  where dct_mat: [n_mfcc, n_mels]
        Tensor dbc = db.contiguous();
        const bool batched = (dbc.dim() == 3);
        const int64_t batch    = batched ? dbc.size(0) : 1;
        const int64_t n_frames = dbc.size(batched ? 2 : 1);

        std::vector<int64_t> out_shape = batched
            ? std::vector<int64_t>{batch, n_mfcc_, n_frames}
            : std::vector<int64_t>{n_mfcc_, n_frames};
        Tensor out = at::zeros(out_shape, at::TensorOptions().dtype(c10::ScalarType::Float));
        float* op = out.mutable_data_ptr<float>();
        const float* dp = dbc.data_ptr<float>();
        const float* dct = dct_matrix_.data_ptr<float>();

        for (int64_t b = 0; b < batch; ++b) {
            const float* dp_b = dp + b * n_mels_ * n_frames;
            float* op_b = op + b * n_mfcc_ * n_frames;
            for (int64_t i = 0; i < n_mfcc_; ++i) {
                for (int64_t t = 0; t < n_frames; ++t) {
                    double acc = 0.0;
                    for (int64_t k = 0; k < n_mels_; ++k) {
                        acc += dct[i * n_mels_ + k] * dp_b[k * n_frames + t];
                    }
                    op_b[i * n_frames + t] = static_cast<float>(acc);
                }
            }
        }
        return out;
    }

private:
    void build_dct_matrix() {
        dct_matrix_ = at::empty({n_mfcc_, n_mels_},
                                at::TensorOptions().dtype(c10::ScalarType::Float));
        float* p = dct_matrix_.mutable_data_ptr<float>();
        const double PI = 3.14159265358979323846;
        const double scale0 = std::sqrt(1.0 / n_mels_);
        const double scale  = std::sqrt(2.0 / n_mels_);
        for (int64_t i = 0; i < n_mfcc_; ++i) {
            for (int64_t k = 0; k < n_mels_; ++k) {
                const double v = std::cos(PI / n_mels_ * (k + 0.5) * i);
                p[i * n_mels_ + k] = static_cast<float>((i == 0 ? scale0 : scale) * v);
            }
        }
    }

    int64_t n_mfcc_, n_mels_;
    MelSpectrogram mel_;
    AmplitudeToDB  to_db_;
    Tensor dct_matrix_;
};

// ---------------------------------------------------------------------------
// Resample module
// ---------------------------------------------------------------------------
class Resample : public Module {
public:
    Resample(int64_t orig_freq = 16000, int64_t new_freq = 16000, int64_t lowpass_filter_width = 6)
        : Module("Resample")
        , orig_freq_(orig_freq)
        , new_freq_(new_freq)
        , width_(lowpass_filter_width)
    {}

    Tensor forward(const Tensor& waveform) override {
        return functional::resample(waveform, orig_freq_, new_freq_, width_);
    }

private:
    int64_t orig_freq_, new_freq_, width_;
};

} // namespace transforms
} // namespace audio
} // namespace torch
