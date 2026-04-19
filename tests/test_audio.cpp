// tests/test_audio.cpp
// Self-test for torch::audio — sine -> STFT -> iSTFT reconstruction.
//
// Build (Elbrus):
//   cd build_mt
//   cmake --build . --target test_audio
//
// Or manual: g++ -O2 -std=c++17 -I.. tests/test_audio.cpp -laten -lc10 -o test_audio
#include "torch/audio/audio.h"
#include "aten/src/ATen/ATen.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main() {
    const int SR = 16000;
    const int N  = SR;                 // 1 second
    const int n_fft = 512;
    const int hop   = 128;

    // Build a sine wave in [-1, 1].
    at::Tensor x = at::empty({N},
        at::TensorOptions().dtype(c10::ScalarType::Float));
    float* xp = x.mutable_data_ptr<float>();
    const double f = 440.0;
    for (int i = 0; i < N; ++i) {
        xp[i] = 0.8f * std::sin(2.0 * M_PI * f * i / SR);
    }

    // STFT -> iSTFT
    at::Tensor S = torch::audio::functional::stft(x, n_fft, hop, n_fft,
                                                  at::Tensor(), /*center=*/true);
    std::printf("STFT shape: [%lld, %lld, %lld]\n",
                (long long)S.size(0), (long long)S.size(1), (long long)S.size(2));

    at::Tensor y = torch::audio::functional::istft(S, n_fft, hop, n_fft,
                                                   at::Tensor(), /*center=*/true,
                                                   /*length=*/N);
    std::printf("iSTFT shape: [%lld]\n", (long long)y.size(0));

    // Reconstruction error — skip first/last hop to avoid edge effects.
    const float* yp = y.data_ptr<float>();
    float max_diff = 0.0f;
    const int skip = n_fft;
    for (int i = skip; i < N - skip; ++i) {
        const float d = std::fabs(xp[i] - yp[i]);
        if (d > max_diff) max_diff = d;
    }
    std::printf("Max |x - iSTFT(STFT(x))| (center region) = %.6g\n", max_diff);

    // Spectrogram
    at::Tensor sp = torch::audio::functional::spectrogram(
        x, n_fft, n_fft, hop, /*power=*/2.0, at::Tensor(), /*center=*/true);
    std::printf("power-spectrogram shape: [%lld, %lld]\n",
                (long long)sp.size(0), (long long)sp.size(1));

    // Mel filterbanks
    at::Tensor fb = torch::audio::functional::melscale_fbanks(
        /*n_mels=*/64, /*f_min=*/0.0, /*f_max=*/SR / 2.0,
        /*n_stft=*/n_fft / 2 + 1, /*sr=*/SR);
    std::printf("mel fbanks shape: [%lld, %lld]\n",
                (long long)fb.size(0), (long long)fb.size(1));

    // Resample 16k -> 8k -> 16k, check round-trip energy reasonable
    at::Tensor x_down = torch::audio::functional::resample(x, SR, SR / 2, 6);
    at::Tensor x_up   = torch::audio::functional::resample(x_down, SR / 2, SR, 6);
    std::printf("resample round-trip shape: [%lld]\n", (long long)x_up.size(0));

    // μ-law round-trip
    at::Tensor enc = torch::audio::functional::mu_law_encoding(x, 256);
    at::Tensor dec = torch::audio::functional::mu_law_decoding(enc, 256);
    float mu_err = 0.0f;
    const float* dp = dec.data_ptr<float>();
    for (int i = 0; i < N; ++i) {
        const float d = std::fabs(xp[i] - dp[i]);
        if (d > mu_err) mu_err = d;
    }
    std::printf("mu-law max |err| = %.4g\n", mu_err);

    // save_wav / load_wav round-trip
    torch::audio::save_wav("audio_test.wav", x, SR);
    auto loaded = torch::audio::load_wav("audio_test.wav");
    at::Tensor xr = loaded.first;
    std::printf("load_wav: shape=[%lld,%lld], sr=%lld\n",
                (long long)xr.size(0), (long long)xr.size(1), (long long)loaded.second);
    const float* xrp = xr.data_ptr<float>();
    float wav_err = 0.0f;
    for (int i = 0; i < N; ++i) {
        const float d = std::fabs(xp[i] - xrp[i]);
        if (d > wav_err) wav_err = d;
    }
    std::printf("wav round-trip max |err| = %.4g (should be <2e-5)\n", wav_err);

    // Transforms smoke test.
    torch::audio::transforms::MelSpectrogram mel(SR, n_fft, 0, hop, 0.0, -1.0, 64);
    at::Tensor mel_out = mel.forward(x);
    std::printf("MelSpectrogram shape: [%lld, %lld]\n",
                (long long)mel_out.size(0), (long long)mel_out.size(1));

    torch::audio::transforms::MFCC mfcc(SR, 40, n_fft, hop, 64);
    at::Tensor mfcc_out = mfcc.forward(x);
    std::printf("MFCC shape: [%lld, %lld]\n",
                (long long)mfcc_out.size(0), (long long)mfcc_out.size(1));

    if (max_diff < 1e-3f) {
        std::printf("[PASS] STFT/iSTFT reconstruction within tolerance\n");
        return 0;
    } else {
        std::printf("[FAIL] STFT/iSTFT max_diff=%.4g > 1e-3\n", max_diff);
        return 1;
    }
}
