#pragma once

// ============================================================================
// PackedSequence utilities for variable-length sequences
// ============================================================================
// Implements:
//   - PackedSequence struct
//   - pack_padded_sequence()
//   - pad_packed_sequence()
//   - pack_sequence()
//   - pad_sequence()
// ============================================================================

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"
#include <vector>
#include <algorithm>
#include <numeric>

namespace torch {
namespace nn {
namespace utils {
namespace rnn {

using at::Tensor;

// ============================================================================
// PackedSequence — packed representation of variable-length sequences
// ============================================================================
// data: 1D tensor of concatenated sequences (sorted by length, descending)
// batch_sizes: 1D tensor where batch_sizes[t] = number of sequences
//              that have length > t
// sorted_indices: original indices before sorting (for unsorting)
// unsorted_indices: inverse permutation of sorted_indices

struct PackedSequence {
    Tensor data;              // [total_elements, *] — packed data
    Tensor batch_sizes;       // [max_seq_len] — batch size at each time step
    Tensor sorted_indices;    // [batch] — indices used for sorting
    Tensor unsorted_indices;  // [batch] — inverse of sorted_indices

    PackedSequence() = default;

    PackedSequence(Tensor data_, Tensor batch_sizes_,
                   Tensor sorted_indices_ = Tensor(),
                   Tensor unsorted_indices_ = Tensor())
        : data(std::move(data_))
        , batch_sizes(std::move(batch_sizes_))
        , sorted_indices(std::move(sorted_indices_))
        , unsorted_indices(std::move(unsorted_indices_))
    {}

    bool defined() const { return data.defined(); }
};

// ============================================================================
// pack_padded_sequence — pack a padded batch of variable-length sequences
// ============================================================================
// input: [T, B, *] or [B, T, *] if batch_first
// lengths: 1D tensor or vector of actual sequence lengths
// batch_first: if true, input is [B, T, *]
// enforce_sorted: if true, check that lengths are sorted descending
//
// Returns: PackedSequence

inline PackedSequence pack_padded_sequence(
    const Tensor& input,
    const std::vector<int64_t>& lengths,
    bool batch_first = false,
    bool enforce_sorted = true)
{
    // Transpose to [T, B, *] if batch_first
    Tensor padded = batch_first ? input.transpose(0, 1).contiguous() : input.contiguous();

    int64_t max_len = padded.size(0);
    int64_t batch_size = padded.size(1);

    PT_CHECK_MSG(static_cast<int64_t>(lengths.size()) == batch_size,
        "pack_padded_sequence: lengths size must equal batch size");

    // Build sorted indices (descending by length)
    std::vector<int64_t> sorted_idx(static_cast<size_t>(batch_size));
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&](int64_t a, int64_t b) { return lengths[static_cast<size_t>(a)] > lengths[static_cast<size_t>(b)]; });

    // Sorted lengths
    std::vector<int64_t> sorted_lengths(static_cast<size_t>(batch_size));
    for (int64_t i = 0; i < batch_size; ++i) {
        sorted_lengths[static_cast<size_t>(i)] = lengths[static_cast<size_t>(sorted_idx[static_cast<size_t>(i)])];
    }

    if (enforce_sorted) {
        // Verify input was already sorted
        for (int64_t i = 0; i < batch_size; ++i) {
            PT_CHECK_MSG(sorted_idx[static_cast<size_t>(i)] == i,
                "pack_padded_sequence: sequences must be sorted by length (descending). "
                "Use enforce_sorted=False to sort automatically.");
        }
    }

    PT_CHECK_MSG(sorted_lengths[0] > 0,
        "pack_padded_sequence: all sequences must have length > 0");
    PT_CHECK_MSG(sorted_lengths[0] <= max_len,
        "pack_padded_sequence: max length exceeds padded dimension");

    // Compute batch_sizes: batch_sizes[t] = number of sequences with length > t
    std::vector<int64_t> batch_sizes_vec(static_cast<size_t>(sorted_lengths[0]));
    for (int64_t t = 0; t < sorted_lengths[0]; ++t) {
        int64_t count = 0;
        for (int64_t b = 0; b < batch_size; ++b) {
            if (sorted_lengths[static_cast<size_t>(b)] > t) {
                ++count;
            }
        }
        batch_sizes_vec[static_cast<size_t>(t)] = count;
    }

    // Total number of elements
    int64_t total = 0;
    for (auto bs : batch_sizes_vec) total += bs;

    // Get feature dimensions (everything after [T, B])
    std::vector<int64_t> feature_sizes;
    for (int64_t d = 2; d < padded.dim(); ++d) {
        feature_sizes.push_back(padded.size(d));
    }

    // Build packed data shape: [total, *feature_sizes]
    std::vector<int64_t> packed_shape = {total};
    for (auto fs : feature_sizes) packed_shape.push_back(fs);

    Tensor packed_data = at::empty(packed_shape);

    // Compute feature stride (product of feature dimensions)
    int64_t feature_numel = 1;
    for (auto fs : feature_sizes) feature_numel *= fs;

    // Fill packed data
    // For each time step t, copy batch_sizes[t] elements from sorted sequences
    int64_t offset = 0;
    const float* src = padded.data_ptr<float>();
    float* dst = packed_data.mutable_data_ptr<float>();

    for (int64_t t = 0; t < sorted_lengths[0]; ++t) {
        int64_t bs = batch_sizes_vec[static_cast<size_t>(t)];
        for (int64_t b = 0; b < bs; ++b) {
            int64_t orig_b = sorted_idx[static_cast<size_t>(b)];
            // Source: padded[t, orig_b, ...]
            int64_t src_offset = (t * batch_size + orig_b) * feature_numel;
            int64_t dst_offset = offset * feature_numel;
            std::memcpy(dst + dst_offset, src + src_offset,
                        static_cast<size_t>(feature_numel) * sizeof(float));
            ++offset;
        }
    }

    // Create batch_sizes tensor (Long)
    Tensor batch_sizes_tensor = at::empty({static_cast<int64_t>(batch_sizes_vec.size())},
                                           at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* bs_ptr = batch_sizes_tensor.mutable_data_ptr<int64_t>();
    for (size_t i = 0; i < batch_sizes_vec.size(); ++i) {
        bs_ptr[i] = batch_sizes_vec[i];
    }

    // Create sorted_indices tensor
    Tensor sorted_indices_tensor = at::empty({batch_size},
                                              at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* si_ptr = sorted_indices_tensor.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i) {
        si_ptr[i] = sorted_idx[static_cast<size_t>(i)];
    }

    // Create unsorted_indices (inverse permutation)
    Tensor unsorted_indices_tensor = at::empty({batch_size},
                                                at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* ui_ptr = unsorted_indices_tensor.mutable_data_ptr<int64_t>();
    for (int64_t i = 0; i < batch_size; ++i) {
        ui_ptr[sorted_idx[static_cast<size_t>(i)]] = i;
    }

    return PackedSequence(packed_data, batch_sizes_tensor,
                          sorted_indices_tensor, unsorted_indices_tensor);
}

// Overload accepting Tensor lengths
inline PackedSequence pack_padded_sequence(
    const Tensor& input,
    const Tensor& lengths,
    bool batch_first = false,
    bool enforce_sorted = true)
{
    PT_CHECK_MSG(lengths.dim() == 1, "pack_padded_sequence: lengths must be 1D");
    Tensor lc = lengths.contiguous();
    std::vector<int64_t> len_vec(static_cast<size_t>(lc.numel()));
    if (lc.dtype() == c10::ScalarType::Long) {
        const int64_t* p = lc.data_ptr<int64_t>();
        for (int64_t i = 0; i < lc.numel(); ++i) len_vec[static_cast<size_t>(i)] = p[i];
    } else if (lc.dtype() == c10::ScalarType::Int) {
        const int32_t* p = lc.data_ptr<int32_t>();
        for (int64_t i = 0; i < lc.numel(); ++i) len_vec[static_cast<size_t>(i)] = static_cast<int64_t>(p[i]);
    } else if (lc.dtype() == c10::ScalarType::Float) {
        const float* p = lc.data_ptr<float>();
        for (int64_t i = 0; i < lc.numel(); ++i) len_vec[static_cast<size_t>(i)] = static_cast<int64_t>(p[i]);
    } else {
        PT_CHECK_MSG(false, "pack_padded_sequence: unsupported lengths dtype");
    }
    return pack_padded_sequence(input, len_vec, batch_first, enforce_sorted);
}

// ============================================================================
// pad_packed_sequence — unpack a PackedSequence back to padded form
// ============================================================================
// sequence: PackedSequence
// batch_first: if true, output is [B, T, *]
// padding_value: value for padded positions
// total_length: pad to this length (if 0, use max length from batch_sizes)
//
// Returns: (padded_output, lengths)
//   padded_output: [T, B, *] or [B, T, *]
//   lengths: 1D tensor of actual sequence lengths (in original order)

inline std::pair<Tensor, Tensor> pad_packed_sequence(
    const PackedSequence& sequence,
    bool batch_first = false,
    float padding_value = 0.0f,
    int64_t total_length = 0)
{
    PT_CHECK_MSG(sequence.defined(), "pad_packed_sequence: PackedSequence is not defined");

    const Tensor& packed_data = sequence.data;
    const Tensor& batch_sizes = sequence.batch_sizes;

    Tensor bs_c = batch_sizes.contiguous();
    const int64_t* bs_ptr = bs_c.data_ptr<int64_t>();
    int64_t max_len = bs_c.numel();
    int64_t batch_size = bs_ptr[0]; // first batch_size is the largest

    if (total_length > 0) {
        PT_CHECK_MSG(total_length >= max_len,
            "pad_packed_sequence: total_length must be >= max sequence length");
        max_len = total_length;
    }

    // Get feature dimensions
    std::vector<int64_t> feature_sizes;
    for (int64_t d = 1; d < packed_data.dim(); ++d) {
        feature_sizes.push_back(packed_data.size(d));
    }
    int64_t feature_numel = 1;
    for (auto fs : feature_sizes) feature_numel *= fs;

    // Output shape: [max_len, batch_size, *feature_sizes]
    std::vector<int64_t> out_shape = {max_len, batch_size};
    for (auto fs : feature_sizes) out_shape.push_back(fs);

    Tensor output = at::full(out_shape, at::Scalar(padding_value));

    // Compute sorted lengths from batch_sizes
    // sorted_lengths[b] = number of time steps where batch_sizes[t] > b
    std::vector<int64_t> sorted_lengths(static_cast<size_t>(batch_size), 0);
    for (int64_t t = 0; t < bs_c.numel(); ++t) {
        for (int64_t b = 0; b < bs_ptr[t]; ++b) {
            sorted_lengths[static_cast<size_t>(b)]++;
        }
    }

    // Fill output from packed data
    const float* src = packed_data.data_ptr<float>();
    float* dst = output.mutable_data_ptr<float>();
    int64_t offset = 0;

    for (int64_t t = 0; t < bs_c.numel(); ++t) {
        int64_t bs = bs_ptr[t];
        for (int64_t b = 0; b < bs; ++b) {
            // Need to map b (sorted index) back to original index
            int64_t orig_b;
            if (sequence.unsorted_indices.defined()) {
                // sorted_indices maps sorted -> original
                // sorted position b => sorted_indices[b] is the original index
                // But we need to write to the sorted position in output,
                // then unsort at the end... OR we unsort here.
                // Actually, the convention: packed data is in sorted order.
                // Output should be in original order if unsorted_indices is available.
                orig_b = sequence.sorted_indices.data_ptr<int64_t>()[b];
            } else {
                orig_b = b;
            }
            int64_t dst_offset = (t * batch_size + orig_b) * feature_numel;
            int64_t src_offset = offset * feature_numel;
            std::memcpy(dst + dst_offset, src + src_offset,
                        static_cast<size_t>(feature_numel) * sizeof(float));
            ++offset;
        }
    }

    // Build lengths tensor in original order
    Tensor lengths_out = at::empty({batch_size},
                                    at::TensorOptions().dtype(c10::ScalarType::Long));
    int64_t* len_ptr = lengths_out.mutable_data_ptr<int64_t>();
    for (int64_t b = 0; b < batch_size; ++b) {
        if (sequence.unsorted_indices.defined()) {
            int64_t orig_b = sequence.sorted_indices.data_ptr<int64_t>()[b];
            len_ptr[orig_b] = sorted_lengths[static_cast<size_t>(b)];
        } else {
            len_ptr[b] = sorted_lengths[static_cast<size_t>(b)];
        }
    }

    if (batch_first) {
        output = output.transpose(0, 1).contiguous();
    }

    return {output, lengths_out};
}

// ============================================================================
// pad_sequence — pad a list of variable-length tensors to the same length
// ============================================================================
// sequences: vector of tensors, each [L_i, *]
// batch_first: if true, output is [B, max_len, *]; else [max_len, B, *]
// padding_value: fill value for padding
//
// Returns: padded tensor

inline Tensor pad_sequence(
    const std::vector<Tensor>& sequences,
    bool batch_first = false,
    float padding_value = 0.0f)
{
    PT_CHECK_MSG(!sequences.empty(), "pad_sequence: empty sequence list");

    int64_t batch_size = static_cast<int64_t>(sequences.size());

    // Find max length
    int64_t max_len = 0;
    for (const auto& seq : sequences) {
        PT_CHECK_MSG(seq.dim() >= 1, "pad_sequence: each sequence must have at least 1 dimension");
        if (seq.size(0) > max_len) max_len = seq.size(0);
    }

    // Get trailing dimensions from first sequence
    std::vector<int64_t> trailing_dims;
    for (int64_t d = 1; d < sequences[0].dim(); ++d) {
        trailing_dims.push_back(sequences[0].size(d));
    }

    // Build output shape: [max_len, batch_size, *trailing_dims]
    std::vector<int64_t> out_shape = {max_len, batch_size};
    for (auto td : trailing_dims) out_shape.push_back(td);

    Tensor output = at::full(out_shape, at::Scalar(padding_value));

    int64_t feature_numel = 1;
    for (auto td : trailing_dims) feature_numel *= td;

    float* dst = output.mutable_data_ptr<float>();

    for (int64_t b = 0; b < batch_size; ++b) {
        Tensor seq = sequences[static_cast<size_t>(b)].contiguous();
        int64_t seq_len = seq.size(0);
        const float* src = seq.data_ptr<float>();
        for (int64_t t = 0; t < seq_len; ++t) {
            int64_t dst_offset = (t * batch_size + b) * feature_numel;
            int64_t src_offset = t * feature_numel;
            std::memcpy(dst + dst_offset, src + src_offset,
                        static_cast<size_t>(feature_numel) * sizeof(float));
        }
    }

    if (batch_first) {
        output = output.transpose(0, 1).contiguous();
    }

    return output;
}

// ============================================================================
// pack_sequence — pack a list of variable-length tensors
// ============================================================================
// sequences: vector of tensors, each [L_i, *], sorted by length descending
// enforce_sorted: if true, verify that sequences are sorted
//
// Returns: PackedSequence

inline PackedSequence pack_sequence(
    const std::vector<Tensor>& sequences,
    bool enforce_sorted = true)
{
    PT_CHECK_MSG(!sequences.empty(), "pack_sequence: empty sequence list");

    // Get lengths
    std::vector<int64_t> lengths(sequences.size());
    for (size_t i = 0; i < sequences.size(); ++i) {
        lengths[i] = sequences[i].size(0);
    }

    // Pad first, then pack
    // pad_sequence gives [max_len, batch, *] with batch_first=false
    Tensor padded = pad_sequence(sequences, /*batch_first=*/false, /*padding_value=*/0.0f);

    return pack_padded_sequence(padded, lengths, /*batch_first=*/false, enforce_sorted);
}

} // namespace rnn
} // namespace utils
} // namespace nn
} // namespace torch
