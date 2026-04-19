// torch/audio/audio.h
// torchaudio-compatible top-level package header for PromeTorch.
//
// Provides:
//   - load_wav(path)             — parse RIFF/WAVE (PCM 8/16/24/32-bit or IEEE float32)
//   - save_wav(path, tensor, sr) — write 16-bit PCM RIFF/WAVE
//
// Also includes torch/audio/functional.h and torch/audio/transforms.h so users
// can `#include "torch/audio/audio.h"` as a single-entry header.
//
// CPU-only, header-only, no external deps.
#pragma once

#include "torch/audio/functional.h"
#include "torch/audio/transforms.h"
#include "aten/src/ATen/ATen.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch {
namespace audio {

using at::Tensor;

namespace detail {

inline uint32_t read_le_u32(const uint8_t* p) {
    return  (uint32_t)p[0]        |
           ((uint32_t)p[1] << 8)  |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}
inline uint16_t read_le_u16(const uint8_t* p) {
    return  (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}
inline void write_le_u16(std::ostream& f, uint16_t v) {
    uint8_t b[2] = { (uint8_t)(v & 0xFF), (uint8_t)((v >> 8) & 0xFF) };
    f.write(reinterpret_cast<char*>(b), 2);
}
inline void write_le_u32(std::ostream& f, uint32_t v) {
    uint8_t b[4] = {
        (uint8_t)(v & 0xFF),
        (uint8_t)((v >> 8)  & 0xFF),
        (uint8_t)((v >> 16) & 0xFF),
        (uint8_t)((v >> 24) & 0xFF),
    };
    f.write(reinterpret_cast<char*>(b), 4);
}

} // namespace detail

// ---------------------------------------------------------------------------
// load_wav
// ---------------------------------------------------------------------------
// Returns {tensor[channels, samples] (float32, normalized to [-1, 1]), sample_rate}.
// Supports:
//   - PCM 8-bit unsigned  (offset 128)
//   - PCM 16-bit signed LE
//   - PCM 24-bit signed LE (packed)
//   - PCM 32-bit signed LE
//   - IEEE float 32
inline std::pair<Tensor, int64_t> load_wav(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("load_wav: cannot open " + path);

    uint8_t hdr[12];
    f.read(reinterpret_cast<char*>(hdr), 12);
    if (!f || std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("load_wav: not a RIFF/WAVE file: " + path);
    }

    uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
    uint32_t sample_rate = 0;
    std::vector<uint8_t> data;

    while (f) {
        uint8_t chunk_hdr[8];
        f.read(reinterpret_cast<char*>(chunk_hdr), 8);
        if (!f) break;
        const uint32_t size = detail::read_le_u32(chunk_hdr + 4);
        if (std::memcmp(chunk_hdr, "fmt ", 4) == 0) {
            std::vector<uint8_t> fmt(size);
            f.read(reinterpret_cast<char*>(fmt.data()), size);
            audio_format    = detail::read_le_u16(fmt.data() + 0);
            num_channels    = detail::read_le_u16(fmt.data() + 2);
            sample_rate     = detail::read_le_u32(fmt.data() + 4);
            bits_per_sample = detail::read_le_u16(fmt.data() + 14);
        } else if (std::memcmp(chunk_hdr, "data", 4) == 0) {
            data.resize(size);
            f.read(reinterpret_cast<char*>(data.data()), size);
            break; // we have what we need
        } else {
            // Skip unknown chunk (incl. LIST, bext, ...).
            f.seekg(size + (size & 1), std::ios::cur);
        }
    }

    if (num_channels == 0 || sample_rate == 0 || bits_per_sample == 0) {
        throw std::runtime_error("load_wav: missing fmt chunk in " + path);
    }
    // audio_format: 1 = PCM, 3 = IEEE float, 0xFFFE = extensible (accept).
    if (audio_format != 1 && audio_format != 3 && audio_format != 0xFFFE) {
        throw std::runtime_error("load_wav: unsupported audio format " +
                                 std::to_string(audio_format));
    }

    const int64_t channels = num_channels;
    const int64_t bytes_per_sample = bits_per_sample / 8;
    const int64_t frame_bytes = bytes_per_sample * channels;
    if (frame_bytes == 0) throw std::runtime_error("load_wav: invalid frame_bytes");
    const int64_t samples = static_cast<int64_t>(data.size()) / frame_bytes;

    Tensor out = at::empty({channels, samples},
                           at::TensorOptions().dtype(c10::ScalarType::Float));
    float* op = out.mutable_data_ptr<float>();

    const uint8_t* d = data.data();
    for (int64_t i = 0; i < samples; ++i) {
        for (int64_t c = 0; c < channels; ++c) {
            const uint8_t* s = d + (i * channels + c) * bytes_per_sample;
            float v = 0.0f;
            if (audio_format == 3 && bits_per_sample == 32) {
                std::memcpy(&v, s, 4);
            } else if (bits_per_sample == 8) {
                // unsigned 8-bit PCM, zero at 128
                v = (static_cast<int>(s[0]) - 128) / 128.0f;
            } else if (bits_per_sample == 16) {
                int16_t x;
                std::memcpy(&x, s, 2);
                v = x / 32768.0f;
            } else if (bits_per_sample == 24) {
                int32_t x = (int32_t)s[0]        |
                           ((int32_t)s[1] << 8) |
                           ((int32_t)s[2] << 16);
                if (x & 0x800000) x |= 0xFF000000; // sign-extend
                v = x / 8388608.0f;
            } else if (bits_per_sample == 32) {
                int32_t x;
                std::memcpy(&x, s, 4);
                v = x / 2147483648.0f;
            } else {
                throw std::runtime_error("load_wav: unsupported bit depth " +
                                         std::to_string(bits_per_sample));
            }
            op[c * samples + i] = v;
        }
    }

    return { out, static_cast<int64_t>(sample_rate) };
}

// ---------------------------------------------------------------------------
// save_wav
// ---------------------------------------------------------------------------
// Writes 16-bit PCM RIFF/WAVE. Accepts [channels, samples] or [samples].
// Values outside [-1, 1] are clamped.
inline void save_wav(const std::string& path, const Tensor& tensor, int64_t sample_rate) {
    Tensor x = tensor.contiguous();
    int64_t channels = 1, samples = 0;
    if (x.dim() == 1) { samples = x.size(0); }
    else if (x.dim() == 2) { channels = x.size(0); samples = x.size(1); }
    else throw std::runtime_error("save_wav: tensor must be 1-D or 2-D");

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("save_wav: cannot open " + path);

    const uint16_t bits_per_sample = 16;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * block_align;
    const uint32_t data_bytes = static_cast<uint32_t>(samples * block_align);
    const uint32_t fmt_chunk_size = 16;
    const uint32_t riff_size = 4 + (8 + fmt_chunk_size) + (8 + data_bytes);

    f.write("RIFF", 4);
    detail::write_le_u32(f, riff_size);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    detail::write_le_u32(f, fmt_chunk_size);
    detail::write_le_u16(f, 1);                                // PCM
    detail::write_le_u16(f, static_cast<uint16_t>(channels));
    detail::write_le_u32(f, static_cast<uint32_t>(sample_rate));
    detail::write_le_u32(f, byte_rate);
    detail::write_le_u16(f, block_align);
    detail::write_le_u16(f, bits_per_sample);

    f.write("data", 4);
    detail::write_le_u32(f, data_bytes);

    const float* src = x.data_ptr<float>();
    for (int64_t i = 0; i < samples; ++i) {
        for (int64_t c = 0; c < channels; ++c) {
            float v = (x.dim() == 1) ? src[i] : src[c * samples + i];
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            int16_t s = static_cast<int16_t>(std::lround(v * 32767.0f));
            uint8_t b[2] = { (uint8_t)(s & 0xFF), (uint8_t)((s >> 8) & 0xFF) };
            f.write(reinterpret_cast<char*>(b), 2);
        }
    }
}

} // namespace audio
} // namespace torch
