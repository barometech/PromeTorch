#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/ReduceOps.h"
#include "aten/src/ATen/native/cpu/MathOps.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"
#include "aten/src/ATen/native/cpu/PromeBLAS.h"
#include <cmath>
#include <map>
#include <algorithm>
#include <numeric>

namespace at {
namespace native {

// ============================================================================
// Matrix-Matrix Multiplication (mm)
// C = A @ B where A is [M, K] and B is [K, N]
// ============================================================================

inline Tensor mm(const Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.dim() == 2, "mm requires 2D tensors, got ", self.dim(), "D");
    PT_CHECK_MSG(other.dim() == 2, "mm requires 2D tensors, got ", other.dim(), "D");
    PT_CHECK_MSG(self.size(1) == other.size(0),
        "mm: mat1 and mat2 shapes cannot be multiplied (",
        self.size(0), "x", self.size(1), " and ",
        other.size(0), "x", other.size(1), ")");

    int64_t M = self.size(0);
    int64_t K = self.size(1);
    int64_t N = other.size(1);

    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());

    // Fast path for float32: detect transposed B to avoid .contiguous() copy
    if (result_dtype == c10::ScalarType::Float) {
        Tensor A = self.contiguous();
        const float* A_data = A.data_ptr<float>();

        Tensor result = at::empty({M, N}, TensorOptions().dtype(result_dtype).device(self.device()));
        float* C = result.mutable_data_ptr<float>();

        // Check if B is a simple 2D transpose: strides [1, K] instead of [N, 1]
        // This means B is a transposed view of original [N, K] matrix
        bool b_is_transpose = (other.stride(0) == 1 && other.stride(1) == other.size(0));

        if (b_is_transpose) {
            // B is transposed view of B_orig[N, K]. Use sgemm_nt to avoid copy.
            const float* B_data = other.data_ptr<float>();
            int64_t ldb = other.stride(1); // = K (leading dim of B_orig)
            blas::sgemm_nt(M, K, N, 1.0f, A_data, K, B_data, ldb, 0.0f, C, N);
        } else {
            Tensor B = other.contiguous();
            const float* B_data = B.data_ptr<float>();
            blas::sgemm(M, K, N, 1.0f, A_data, K, B_data, N, 0.0f, C, N);
        }

        return result;
    }

    // Non-float fallback: make contiguous and use scalar loop
    Tensor A = self.contiguous();
    Tensor B = other.contiguous();
    Tensor result = zeros({M, N}, TensorOptions().dtype(result_dtype).device(A.device()));

    PT_DISPATCH_FLOATING_TYPES(result_dtype, "mm", [&] {
        const scalar_t* A_data = A.data_ptr<scalar_t>();
        const scalar_t* B_data = B.data_ptr<scalar_t>();
        scalar_t* C = result.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                scalar_t sum = 0;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A_data[i * K + k] * B_data[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    });

    return result;
}

// ============================================================================
// Matrix-Vector Multiplication (mv)
// y = A @ x where A is [M, N] and x is [N]
// ============================================================================

inline Tensor mv(const Tensor& self, const Tensor& vec) {
    PT_CHECK_MSG(self.dim() == 2, "mv requires matrix to be 2D");
    PT_CHECK_MSG(vec.dim() == 1, "mv requires vector to be 1D");
    PT_CHECK_MSG(self.size(1) == vec.size(0),
        "mv: matrix and vector shapes cannot be multiplied");

    // Make tensors contiguous for correct memory access
    Tensor A = self.contiguous();
    Tensor x = vec.contiguous();

    int64_t M = A.size(0);
    int64_t N = A.size(1);

    c10::ScalarType result_dtype = c10::promoteTypes(A.dtype(), x.dtype());
    Tensor result = zeros({M}, TensorOptions().dtype(result_dtype).device(A.device()));

    if (result_dtype == c10::ScalarType::Float) {
        blas::sgemv(M, N, 1.0f, A.data_ptr<float>(), N, x.data_ptr<float>(),
                    0.0f, result.mutable_data_ptr<float>());
    } else {
        PT_DISPATCH_FLOATING_TYPES(result_dtype, "mv", [&] {
            const scalar_t* A_data = A.data_ptr<scalar_t>();
            const scalar_t* x_data = x.data_ptr<scalar_t>();
            scalar_t* y = result.mutable_data_ptr<scalar_t>();
            for (int64_t i = 0; i < M; ++i) {
                scalar_t sum = 0;
                for (int64_t j = 0; j < N; ++j) {
                    sum += A_data[i * N + j] * x_data[j];
                }
                y[i] = sum;
            }
        });
    }

    return result;
}

// ============================================================================
// Batched Matrix Multiplication (bmm)
// C = A @ B where A is [B, M, K] and B is [B, K, N]
// ============================================================================

inline Tensor bmm(const Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.dim() == 3, "bmm requires 3D tensors");
    PT_CHECK_MSG(other.dim() == 3, "bmm requires 3D tensors");
    PT_CHECK_MSG(self.size(0) == other.size(0),
        "bmm: batch dimensions must match");
    PT_CHECK_MSG(self.size(2) == other.size(1),
        "bmm: matrix dimensions cannot be multiplied");

    // Make tensors contiguous for correct memory access
    Tensor A = self.contiguous();
    Tensor B = other.contiguous();

    int64_t batch = A.size(0);
    int64_t M = A.size(1);
    int64_t K = A.size(2);
    int64_t N = B.size(2);

    c10::ScalarType result_dtype = c10::promoteTypes(A.dtype(), B.dtype());
    Tensor result = zeros({batch, M, N}, TensorOptions().dtype(result_dtype).device(A.device()));

    if (result_dtype == c10::ScalarType::Float) {
        const float* A_data = A.data_ptr<float>();
        const float* B_data = B.data_ptr<float>();
        float* C = result.mutable_data_ptr<float>();
        for (int64_t b = 0; b < batch; ++b) {
            blas::sgemm(M, K, N, 1.0f,
                        A_data + b * M * K, K,
                        B_data + b * K * N, N,
                        0.0f, C + b * M * N, N);
        }
    } else {
        PT_DISPATCH_FLOATING_TYPES(result_dtype, "bmm", [&] {
            const scalar_t* A_data = A.data_ptr<scalar_t>();
            const scalar_t* B_data = B.data_ptr<scalar_t>();
            scalar_t* C = result.mutable_data_ptr<scalar_t>();
            for (int64_t b = 0; b < batch; ++b) {
                const scalar_t* A_b = A_data + b * M * K;
                const scalar_t* B_b = B_data + b * K * N;
                scalar_t* C_b = C + b * M * N;
                for (int64_t i = 0; i < M; ++i) {
                    for (int64_t j = 0; j < N; ++j) {
                        scalar_t sum = 0;
                        for (int64_t k = 0; k < K; ++k) {
                            sum += A_b[i * K + k] * B_b[k * N + j];
                        }
                        C_b[i * N + j] = sum;
                    }
                }
            }
        });
    }

    return result;
}

// ============================================================================
// Dot Product
// ============================================================================

inline Tensor dot(const Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.dim() == 1, "dot requires 1D tensors");
    PT_CHECK_MSG(other.dim() == 1, "dot requires 1D tensors");
    PT_CHECK_MSG(self.size(0) == other.size(0),
        "dot: vectors must have same size");

    // Make tensors contiguous for correct memory access
    Tensor a = self.contiguous();
    Tensor b = other.contiguous();

    int64_t n = a.size(0);
    c10::ScalarType result_dtype = c10::promoteTypes(a.dtype(), b.dtype());
    Tensor result = zeros({}, TensorOptions().dtype(result_dtype).device(a.device()));

    if (result_dtype == c10::ScalarType::Float) {
        float val = blas::sdot(n, a.data_ptr<float>(), b.data_ptr<float>());
        result.mutable_data_ptr<float>()[0] = val;
    } else {
        PT_DISPATCH_FLOATING_TYPES(result_dtype, "dot", [&] {
            const scalar_t* a_data = a.data_ptr<scalar_t>();
            const scalar_t* b_data = b.data_ptr<scalar_t>();
            scalar_t sum = 0;
            for (int64_t i = 0; i < n; ++i) sum += a_data[i] * b_data[i];
            result.mutable_data_ptr<scalar_t>()[0] = sum;
        });
    }

    return result;
}

// ============================================================================
// General Matrix Multiplication (matmul)
// Handles all tensor dimensions with broadcasting
// ============================================================================

inline Tensor matmul(const Tensor& self, const Tensor& other) {
    int64_t dim_self = self.dim();
    int64_t dim_other = other.dim();

    // Vector @ Vector -> scalar
    if (dim_self == 1 && dim_other == 1) {
        return dot(self, other);
    }

    // Matrix @ Vector -> Vector
    if (dim_self == 2 && dim_other == 1) {
        return mv(self, other);
    }

    // Vector @ Matrix -> Vector (1xN @ NxM = M)
    if (dim_self == 1 && dim_other == 2) {
        // Treat vector as 1xN matrix
        Tensor result = mm(self.unsqueeze(0), other);
        return result.squeeze(0);
    }

    // Matrix @ Matrix -> Matrix
    if (dim_self == 2 && dim_other == 2) {
        return mm(self, other);
    }

    // Batched matrix multiplication
    if (dim_self >= 3 && dim_other >= 3) {
        // Handle broadcasting for batch dimensions
        // For simplicity, require same batch dimensions
        PT_CHECK_MSG(dim_self == dim_other,
            "matmul: for batched tensors, dimensions must match");

        if (dim_self == 3) {
            return bmm(self, other);
        }

        // For higher dimensions, reshape to 3D, do bmm, reshape back
        // Simplified: flatten batch dimensions
        int64_t batch = 1;
        for (int64_t i = 0; i < dim_self - 2; ++i) {
            PT_CHECK_MSG(self.size(i) == other.size(i),
                "matmul: batch dimensions must match");
            batch *= self.size(i);
        }

        int64_t M = self.size(-2);
        int64_t K = self.size(-1);
        int64_t N = other.size(-1);

        PT_CHECK_MSG(K == other.size(-2),
            "matmul: matrix dimensions cannot be multiplied");

        // Reshape to 3D
        Tensor self_3d = self.reshape({batch, M, K});
        Tensor other_3d = other.reshape({batch, K, N});

        Tensor result_3d = bmm(self_3d, other_3d);

        // Reshape back
        std::vector<int64_t> result_shape;
        for (int64_t i = 0; i < dim_self - 2; ++i) {
            result_shape.push_back(self.size(i));
        }
        result_shape.push_back(M);
        result_shape.push_back(N);

        return result_3d.reshape(result_shape);
    }

    // Batched @ 2D or 2D @ Batched
    if (dim_self >= 3 && dim_other == 2) {
        // Broadcast other to batch dimensions
        int64_t batch = 1;
        for (int64_t i = 0; i < dim_self - 2; ++i) {
            batch *= self.size(i);
        }

        int64_t M = self.size(-2);
        int64_t K = self.size(-1);
        int64_t N = other.size(-1);

        PT_CHECK_MSG(K == other.size(0),
            "matmul: matrix dimensions cannot be multiplied");

        Tensor self_3d = self.reshape({batch, M, K});
        Tensor other_3d = other.unsqueeze(0).expand({batch, K, N});

        Tensor result_3d = bmm(self_3d, other_3d);

        std::vector<int64_t> result_shape;
        for (int64_t i = 0; i < dim_self - 2; ++i) {
            result_shape.push_back(self.size(i));
        }
        result_shape.push_back(M);
        result_shape.push_back(N);

        return result_3d.reshape(result_shape);
    }

    if (dim_self == 2 && dim_other >= 3) {
        int64_t batch = 1;
        for (int64_t i = 0; i < dim_other - 2; ++i) {
            batch *= other.size(i);
        }

        int64_t M = self.size(0);
        int64_t K = self.size(1);
        int64_t N = other.size(-1);

        PT_CHECK_MSG(K == other.size(-2),
            "matmul: matrix dimensions cannot be multiplied");

        Tensor self_3d = self.unsqueeze(0).expand({batch, M, K});
        Tensor other_3d = other.reshape({batch, K, N});

        Tensor result_3d = bmm(self_3d, other_3d);

        std::vector<int64_t> result_shape;
        for (int64_t i = 0; i < dim_other - 2; ++i) {
            result_shape.push_back(other.size(i));
        }
        result_shape.push_back(M);
        result_shape.push_back(N);

        return result_3d.reshape(result_shape);
    }

    PT_ERROR("matmul: unsupported tensor dimensions");
    return Tensor();  // Unreachable
}

// ============================================================================
// Outer Product
// ============================================================================

inline Tensor outer(const Tensor& self, const Tensor& other) {
    PT_CHECK_MSG(self.dim() == 1, "outer requires 1D tensors");
    PT_CHECK_MSG(other.dim() == 1, "outer requires 1D tensors");

    // Make tensors contiguous for correct memory access
    Tensor a = self.contiguous();
    Tensor b = other.contiguous();

    int64_t M = a.size(0);
    int64_t N = b.size(0);

    c10::ScalarType result_dtype = c10::promoteTypes(a.dtype(), b.dtype());
    Tensor result = empty({M, N}, TensorOptions().dtype(result_dtype).device(a.device()));

    PT_DISPATCH_ALL_TYPES(result_dtype, "outer", [&] {
        const scalar_t* a_data = a.data_ptr<scalar_t>();
        const scalar_t* b_data = b.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                out[i * N + j] = a_data[i] * b_data[j];
            }
        }
    });

    return result;
}

// ============================================================================
// Addmm: C = beta * C + alpha * A @ B
// ============================================================================

inline Tensor addmm(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    Scalar beta = 1,
    Scalar alpha = 1
) {
    PT_CHECK_MSG(self.dim() <= 2, "addmm: self must be at most 2D");
    PT_CHECK_MSG(mat1.dim() == 2, "addmm: mat1 must be 2D");
    PT_CHECK_MSG(mat2.dim() == 2, "addmm: mat2 must be 2D");

    int64_t M = mat1.size(0);
    int64_t K = mat1.size(1);
    int64_t N = mat2.size(1);

    PT_CHECK_MSG(K == mat2.size(0), "addmm: mat1 and mat2 shapes cannot be multiplied");

    // Broadcast self to result shape
    Tensor result = self.expand({M, N}).clone();

    if (result.dtype() == c10::ScalarType::Float) {
        Tensor A = mat1.contiguous();
        float beta_val = beta.to<float>();
        float alpha_val = alpha.to<float>();
        const float* A_data = A.data_ptr<float>();
        float* C = result.mutable_data_ptr<float>();

        // Check if B is a simple 2D transpose
        bool b_is_transpose = (mat2.stride(0) == 1 && mat2.stride(1) == mat2.size(0));
        if (b_is_transpose) {
            blas::sgemm_nt(M, K, N, alpha_val, A_data, K,
                           mat2.data_ptr<float>(), mat2.stride(1),
                           beta_val, C, N);
        } else {
            Tensor B = mat2.contiguous();
            blas::sgemm(M, K, N, alpha_val, A_data, K,
                        B.data_ptr<float>(), N,
                        beta_val, C, N);
        }
    } else {
        Tensor A = mat1.contiguous();
        Tensor B = mat2.contiguous();
        PT_DISPATCH_FLOATING_TYPES(result.dtype(), "addmm", [&] {
            scalar_t beta_val = beta.to<scalar_t>();
            scalar_t alpha_val = alpha.to<scalar_t>();
            const scalar_t* A_data = A.data_ptr<scalar_t>();
            const scalar_t* B_data = B.data_ptr<scalar_t>();
            scalar_t* C = result.mutable_data_ptr<scalar_t>();
            int64_t total = M * N;
            for (int64_t i = 0; i < total; ++i) C[i] *= beta_val;
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    scalar_t sum = 0;
                    for (int64_t k = 0; k < K; ++k) sum += A_data[i * K + k] * B_data[k * N + j];
                    C[i * N + j] += alpha_val * sum;
                }
            }
        });
    }

    return result;
}

// ============================================================================
// Einstein Summation (einsum)
// ============================================================================

namespace einsum_detail {

struct EinsumParsed {
    std::vector<std::string> input_subscripts;
    std::string output_subscript;
    std::vector<char> all_labels;           // unique labels in order
    std::vector<char> contraction_labels;   // labels NOT in output
    std::vector<char> free_labels;          // labels IN output
};

inline EinsumParsed parse_einsum(const std::string& equation) {
    EinsumParsed result;

    // Split on "->"
    size_t arrow = equation.find("->");
    std::string lhs, rhs;
    if (arrow != std::string::npos) {
        lhs = equation.substr(0, arrow);
        rhs = equation.substr(arrow + 2);
    } else {
        lhs = equation;
        // Implicit output: sorted free indices
        // (indices that appear exactly once)
        rhs = ""; // will compute below
    }

    // Remove spaces
    lhs.erase(std::remove(lhs.begin(), lhs.end(), ' '), lhs.end());
    rhs.erase(std::remove(rhs.begin(), rhs.end(), ' '), rhs.end());

    // Split input subscripts on ","
    std::string current;
    for (char c : lhs) {
        if (c == ',') {
            result.input_subscripts.push_back(current);
            current.clear();
        } else {
            current += c;
        }
    }
    result.input_subscripts.push_back(current);

    // Collect all unique labels
    std::map<char, int> label_count;
    for (const auto& sub : result.input_subscripts) {
        for (char c : sub) {
            label_count[c]++;
            if (std::find(result.all_labels.begin(), result.all_labels.end(), c) == result.all_labels.end()) {
                result.all_labels.push_back(c);
            }
        }
    }

    // Compute implicit output if needed
    if (arrow == std::string::npos) {
        std::vector<char> sorted_labels;
        for (char c : result.all_labels) {
            if (label_count[c] == 1) {
                sorted_labels.push_back(c);
            }
        }
        std::sort(sorted_labels.begin(), sorted_labels.end());
        rhs = std::string(sorted_labels.begin(), sorted_labels.end());
    }

    result.output_subscript = rhs;

    // Contraction labels = all_labels - output labels
    for (char c : result.all_labels) {
        if (rhs.find(c) != std::string::npos) {
            result.free_labels.push_back(c);
        } else {
            result.contraction_labels.push_back(c);
        }
    }

    return result;
}

} // namespace einsum_detail

inline Tensor einsum(const std::string& equation, const std::vector<Tensor>& tensors) {
    auto parsed = einsum_detail::parse_einsum(equation);
    PT_CHECK_MSG(parsed.input_subscripts.size() == tensors.size(),
        "einsum: number of subscripts (", parsed.input_subscripts.size(),
        ") does not match number of tensors (", tensors.size(), ")");

    // === OPTIMIZED PATHS ===

    // Two-operand operations
    if (tensors.size() == 2) {
        const auto& sub0 = parsed.input_subscripts[0];
        const auto& sub1 = parsed.input_subscripts[1];
        const auto& out = parsed.output_subscript;

        // ij,jk->ik  (matrix multiply)
        if (sub0 == "ij" && sub1 == "jk" && out == "ik") {
            return mm(tensors[0], tensors[1]);
        }
        // ij,kj->ik  (mm with transpose)
        if (sub0 == "ij" && sub1 == "kj" && out == "ik") {
            return mm(tensors[0], t(tensors[1]));
        }
        // ji,jk->ik  (transpose mm)
        if (sub0 == "ji" && sub1 == "jk" && out == "ik") {
            return mm(t(tensors[0]), tensors[1]);
        }
        // bij,bjk->bik  (batched mm)
        if (sub0 == "bij" && sub1 == "bjk" && out == "bik") {
            return bmm(tensors[0], tensors[1]);
        }
        // ij,j->i  (matrix-vector)
        if (sub0 == "ij" && sub1 == "j" && out == "i") {
            return mv(tensors[0], tensors[1]);
        }
        // i,j->ij  (outer product)
        if (sub0 == "i" && sub1 == "j" && out == "ij") {
            return outer(tensors[0], tensors[1]);
        }
        // i,i->  (dot product)
        if (sub0 == "i" && sub1 == "i" && out == "") {
            return dot(tensors[0], tensors[1]);
        }
    }

    // Single operand
    if (tensors.size() == 1) {
        const auto& sub0 = parsed.input_subscripts[0];
        const auto& out = parsed.output_subscript;

        // ij->ji  (transpose)
        if (sub0 == "ij" && out == "ji") {
            return t(tensors[0]);
        }
        // ij->j  (sum over first dim)
        if (sub0 == "ij" && out == "j") {
            return at::native::sum(tensors[0], 0);
        }
        // ij->i  (sum over second dim)
        if (sub0 == "ij" && out == "i") {
            return at::native::sum(tensors[0], 1);
        }
        // ij->  (sum all)
        if (sub0 == "ij" && out == "") {
            return at::native::sum(tensors[0]);
        }
        // ii->i  (diagonal)
        if (sub0 == "ii" && out == "i") {
            PT_CHECK(tensors[0].dim() == 2 && tensors[0].size(0) == tensors[0].size(1));
            return at::native::diag(tensors[0], 0);
        }
        // ii->  (trace)
        if (sub0 == "ii" && out == "") {
            PT_CHECK(tensors[0].dim() == 2 && tensors[0].size(0) == tensors[0].size(1));
            return at::native::sum(at::native::diag(tensors[0], 0));
        }
    }

    // === GENERAL PATH via permute + reshape + mm ===
    // For two operands: permute both so contraction dims are at end/beginning,
    // reshape to 2D, mm, reshape back, permute to output order

    if (tensors.size() == 2) {
        const auto& sub0 = parsed.input_subscripts[0];
        const auto& sub1 = parsed.input_subscripts[1];
        const auto& out = parsed.output_subscript;

        // Build dimension maps: label -> (tensor_idx, dim_idx)
        std::map<char, int64_t> label_to_dim0, label_to_dim1;
        std::map<char, int64_t> label_to_size;

        for (size_t i = 0; i < sub0.size(); ++i) {
            label_to_dim0[sub0[i]] = i;
            label_to_size[sub0[i]] = tensors[0].size(i);
        }
        for (size_t i = 0; i < sub1.size(); ++i) {
            label_to_dim1[sub1[i]] = i;
            label_to_size[sub1[i]] = tensors[1].size(i);
        }

        // Classify labels
        std::vector<char> free0_labels, free1_labels, contract_labels;
        for (char c : sub0) {
            if (sub1.find(c) != std::string::npos) {
                if (std::find(contract_labels.begin(), contract_labels.end(), c) == contract_labels.end()) {
                    if (out.find(c) == std::string::npos) {
                        contract_labels.push_back(c);
                    }
                }
            }
        }
        for (char c : sub0) {
            if (std::find(contract_labels.begin(), contract_labels.end(), c) == contract_labels.end()) {
                if (std::find(free0_labels.begin(), free0_labels.end(), c) == free0_labels.end()) {
                    free0_labels.push_back(c);
                }
            }
        }
        for (char c : sub1) {
            if (std::find(contract_labels.begin(), contract_labels.end(), c) == contract_labels.end()) {
                if (std::find(free1_labels.begin(), free1_labels.end(), c) == free1_labels.end()) {
                    free1_labels.push_back(c);
                }
            }
        }

        // Permute tensor0: [free0..., contract...]
        std::vector<int64_t> perm0;
        int64_t free0_numel = 1, contract_numel = 1;
        for (char c : free0_labels) {
            perm0.push_back(label_to_dim0[c]);
            free0_numel *= label_to_size[c];
        }
        for (char c : contract_labels) {
            perm0.push_back(label_to_dim0[c]);
            contract_numel *= label_to_size[c];
        }

        // Permute tensor1: [contract..., free1...]
        std::vector<int64_t> perm1;
        int64_t free1_numel = 1;
        for (char c : contract_labels) {
            perm1.push_back(label_to_dim1[c]);
        }
        for (char c : free1_labels) {
            perm1.push_back(label_to_dim1[c]);
            free1_numel *= label_to_size[c];
        }

        Tensor t0 = permute(tensors[0], perm0).contiguous().reshape({free0_numel, contract_numel});
        Tensor t1 = permute(tensors[1], perm1).contiguous().reshape({contract_numel, free1_numel});

        Tensor mm_result = mm(t0, t1);  // [free0_numel, free1_numel]

        // Build result shape from free0 + free1 labels
        std::vector<int64_t> result_shape;
        std::string result_labels;
        for (char c : free0_labels) {
            result_shape.push_back(label_to_size[c]);
            result_labels += c;
        }
        for (char c : free1_labels) {
            result_shape.push_back(label_to_size[c]);
            result_labels += c;
        }

        Tensor result = mm_result.reshape(result_shape);

        // Permute to output order if different
        if (result_labels != out && !out.empty()) {
            std::vector<int64_t> final_perm;
            for (char c : out) {
                auto pos = result_labels.find(c);
                PT_CHECK_MSG(pos != std::string::npos, "einsum: output label '", c, "' not found");
                final_perm.push_back(static_cast<int64_t>(pos));
            }
            result = permute(result, final_perm);
        }

        return result;
    }

    PT_CHECK_MSG(false, "einsum: general case with ", tensors.size(), " operands not yet supported (use 1 or 2 operands)");
    return Tensor();
}

// ============================================================================
// LU Decomposition with partial pivoting
// Returns L (lower triangular, unit diagonal), U (upper triangular), P (permutation)
// ============================================================================

struct LUResult {
    Tensor L, U, P;
};

inline LUResult lu(const Tensor& self) {
    PT_CHECK_MSG(self.dim() == 2, "lu requires 2D tensor");
    PT_CHECK_MSG(self.size(0) == self.size(1), "lu requires square matrix");

    Tensor A = self.contiguous().clone();
    int64_t n = A.size(0);

    // Initialize permutation as identity
    std::vector<int64_t> perm(n);
    std::iota(perm.begin(), perm.end(), 0);

    PT_DISPATCH_FLOATING_TYPES(A.dtype(), "lu", [&] {
        scalar_t* data = A.mutable_data_ptr<scalar_t>();

        for (int64_t k = 0; k < n; ++k) {
            // Find pivot
            int64_t max_row = k;
            scalar_t max_val = std::abs(data[k * n + k]);
            for (int64_t i = k + 1; i < n; ++i) {
                scalar_t val = std::abs(data[i * n + k]);
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }

            // Swap rows
            if (max_row != k) {
                std::swap(perm[k], perm[max_row]);
                for (int64_t j = 0; j < n; ++j) {
                    std::swap(data[k * n + j], data[max_row * n + j]);
                }
            }

            // Check for singularity
            if (std::abs(data[k * n + k]) < 1e-12) continue;

            // Eliminate below
            for (int64_t i = k + 1; i < n; ++i) {
                data[i * n + k] /= data[k * n + k];
                for (int64_t j = k + 1; j < n; ++j) {
                    data[i * n + j] -= data[i * n + k] * data[k * n + j];
                }
            }
        }
    });

    // Extract L, U, P
    Tensor L = eye(n, n, TensorOptions().dtype(self.dtype()));
    Tensor U = zeros({n, n}, TensorOptions().dtype(self.dtype()));
    Tensor P = zeros({n, n}, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "lu_extract", [&] {
        const scalar_t* data = A.data_ptr<scalar_t>();
        scalar_t* L_data = L.mutable_data_ptr<scalar_t>();
        scalar_t* U_data = U.mutable_data_ptr<scalar_t>();
        scalar_t* P_data = P.mutable_data_ptr<scalar_t>();

        for (int64_t i = 0; i < n; ++i) {
            P_data[i * n + perm[i]] = static_cast<scalar_t>(1);
            for (int64_t j = 0; j < n; ++j) {
                if (j < i) {
                    L_data[i * n + j] = data[i * n + j];
                } else {
                    U_data[i * n + j] = data[i * n + j];
                }
            }
        }
    });

    return {L, U, P};
}

// ============================================================================
// Matrix Inverse via LU decomposition
// A^{-1} = solve(A, I)
// ============================================================================

inline Tensor inverse(const Tensor& self) {
    PT_CHECK_MSG(self.dim() == 2, "inverse requires 2D tensor");
    PT_CHECK_MSG(self.size(0) == self.size(1), "inverse requires square matrix");

    int64_t n = self.size(0);
    auto lu_r_ = lu(self);
    Tensor L = lu_r_.L; Tensor U = lu_r_.U; Tensor P = lu_r_.P;

    Tensor result = zeros({n, n}, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "inverse", [&] {
        const scalar_t* L_data = L.data_ptr<scalar_t>();
        const scalar_t* U_data = U.data_ptr<scalar_t>();
        const scalar_t* P_data = P.data_ptr<scalar_t>();
        scalar_t* res_data = result.mutable_data_ptr<scalar_t>();

        // Solve AX = I column by column
        // PA = LU, so AX = I => LU X = P I = P
        for (int64_t col = 0; col < n; ++col) {
            // b = P * e_col
            std::vector<scalar_t> b(n, 0);
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (j == col) b[i] += P_data[i * n + j];
                }
            }

            // Forward substitution: Ly = b
            std::vector<scalar_t> y(n);
            for (int64_t i = 0; i < n; ++i) {
                scalar_t sum = b[i];
                for (int64_t j = 0; j < i; ++j) {
                    sum -= L_data[i * n + j] * y[j];
                }
                y[i] = sum; // L has unit diagonal
            }

            // Backward substitution: Ux = y
            std::vector<scalar_t> x(n);
            for (int64_t i = n - 1; i >= 0; --i) {
                scalar_t sum = y[i];
                for (int64_t j = i + 1; j < n; ++j) {
                    sum -= U_data[i * n + j] * x[j];
                }
                x[i] = sum / U_data[i * n + i];
            }

            for (int64_t i = 0; i < n; ++i) {
                res_data[i * n + col] = x[i];
            }
        }
    });

    return result;
}

// ============================================================================
// Solve: solve(A, b) -> x such that Ax = b
// ============================================================================

inline Tensor solve(const Tensor& A, const Tensor& b) {
    PT_CHECK_MSG(A.dim() == 2, "solve requires 2D matrix A");
    PT_CHECK_MSG(A.size(0) == A.size(1), "solve requires square matrix A");

    int64_t n = A.size(0);
    bool is_vector = (b.dim() == 1);

    Tensor B = is_vector ? b.unsqueeze(1) : b;
    PT_CHECK_MSG(B.size(0) == n, "solve: dimensions mismatch");

    int64_t nrhs = B.size(1);
    auto lu_ra_ = lu(A);
    Tensor L = lu_ra_.L; Tensor U = lu_ra_.U; Tensor P = lu_ra_.P;

    Tensor result = zeros({n, nrhs}, TensorOptions().dtype(A.dtype()));

    PT_DISPATCH_FLOATING_TYPES(A.dtype(), "solve", [&] {
        const scalar_t* L_data = L.data_ptr<scalar_t>();
        const scalar_t* U_data = U.data_ptr<scalar_t>();
        const scalar_t* P_data = P.data_ptr<scalar_t>();
        Tensor B_contig = B.contiguous();
        const scalar_t* B_data = B_contig.data_ptr<scalar_t>();
        scalar_t* res_data = result.mutable_data_ptr<scalar_t>();

        for (int64_t col = 0; col < nrhs; ++col) {
            // Pb
            std::vector<scalar_t> pb(n, 0);
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    pb[i] += P_data[i * n + j] * B_data[j * nrhs + col];
                }
            }

            // Forward substitution: Ly = Pb
            std::vector<scalar_t> y(n);
            for (int64_t i = 0; i < n; ++i) {
                scalar_t s = pb[i];
                for (int64_t j = 0; j < i; ++j) s -= L_data[i * n + j] * y[j];
                y[i] = s;
            }

            // Backward substitution: Ux = y
            std::vector<scalar_t> x(n);
            for (int64_t i = n - 1; i >= 0; --i) {
                scalar_t s = y[i];
                for (int64_t j = i + 1; j < n; ++j) s -= U_data[i * n + j] * x[j];
                x[i] = s / U_data[i * n + i];
            }

            for (int64_t i = 0; i < n; ++i) {
                res_data[i * nrhs + col] = x[i];
            }
        }
    });

    return is_vector ? result.squeeze(1) : result;
}

// ============================================================================
// Determinant via LU decomposition
// det(A) = sign(P) * prod(diag(U))
// ============================================================================

inline Tensor det(const Tensor& self) {
    PT_CHECK_MSG(self.dim() == 2, "det requires 2D tensor");
    PT_CHECK_MSG(self.size(0) == self.size(1), "det requires square matrix");

    auto lu_r_ = lu(self);
    Tensor L = lu_r_.L; Tensor U = lu_r_.U; Tensor P = lu_r_.P;
    int64_t n = self.size(0);

    Tensor result = zeros({}, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "det", [&] {
        const scalar_t* U_data = U.data_ptr<scalar_t>();
        const scalar_t* P_data = P.data_ptr<scalar_t>();

        // Product of diagonal of U
        scalar_t prod = 1;
        for (int64_t i = 0; i < n; ++i) {
            prod *= U_data[i * n + i];
        }

        // Sign of permutation: count swaps
        // Extract permutation vector from P matrix
        std::vector<int64_t> perm(n);
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                if (static_cast<double>(P_data[i * n + j]) > 0.5) {
                    perm[i] = j;
                    break;
                }
            }
        }

        // Count inversions (number of swaps)
        int swaps = 0;
        std::vector<bool> visited(n, false);
        for (int64_t i = 0; i < n; ++i) {
            if (!visited[i]) {
                int64_t j = i;
                int cycle_len = 0;
                while (!visited[j]) {
                    visited[j] = true;
                    j = perm[j];
                    cycle_len++;
                }
                swaps += cycle_len - 1;
            }
        }

        scalar_t sign = (swaps % 2 == 0) ? 1 : -1;
        result.mutable_data_ptr<scalar_t>()[0] = sign * prod;
    });

    return result;
}

// ============================================================================
// Cholesky decomposition: A = L @ L^T (for symmetric positive-definite matrices)
// ============================================================================

inline Tensor cholesky(const Tensor& self, bool upper = false) {
    PT_CHECK_MSG(self.dim() == 2, "cholesky requires 2D tensor");
    PT_CHECK_MSG(self.size(0) == self.size(1), "cholesky requires square matrix");

    int64_t n = self.size(0);
    Tensor L = zeros({n, n}, TensorOptions().dtype(self.dtype()));
    Tensor A = self.contiguous();

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "cholesky", [&] {
        const scalar_t* A_data = A.data_ptr<scalar_t>();
        scalar_t* L_data = L.mutable_data_ptr<scalar_t>();

        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j <= i; ++j) {
                scalar_t sum = 0;
                for (int64_t k = 0; k < j; ++k) {
                    sum += L_data[i * n + k] * L_data[j * n + k];
                }

                if (i == j) {
                    scalar_t val = A_data[i * n + i] - sum;
                    PT_CHECK_MSG(val > 0, "cholesky: matrix is not positive definite");
                    L_data[i * n + j] = std::sqrt(val);
                } else {
                    L_data[i * n + j] = (A_data[i * n + j] - sum) / L_data[j * n + j];
                }
            }
        }
    });

    return upper ? t(L) : L;
}

// ============================================================================
// QR decomposition via Householder reflections
// A = Q @ R
// ============================================================================

struct QRResult {
    Tensor Q, R;
};

inline QRResult qr(const Tensor& self) {
    PT_CHECK_MSG(self.dim() == 2, "qr requires 2D tensor");

    int64_t m = self.size(0);
    int64_t n = self.size(1);
    int64_t k = std::min(m, n);

    Tensor R = self.contiguous().clone();
    Tensor Q = eye(m, m, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "qr", [&] {
        scalar_t* R_data = R.mutable_data_ptr<scalar_t>();
        scalar_t* Q_data = Q.mutable_data_ptr<scalar_t>();

        for (int64_t j = 0; j < k; ++j) {
            // Compute Householder vector
            std::vector<scalar_t> v(m - j);
            scalar_t norm = 0;
            for (int64_t i = j; i < m; ++i) {
                v[i - j] = R_data[i * n + j];
                norm += v[i - j] * v[i - j];
            }
            norm = std::sqrt(norm);

            if (norm < 1e-15) continue;

            scalar_t sign = (v[0] >= 0) ? 1 : -1;
            v[0] += sign * norm;

            // Normalize v
            scalar_t v_norm = 0;
            for (auto& vi : v) v_norm += vi * vi;
            if (v_norm < 1e-30) continue;

            // Apply H = I - 2*v*v^T/||v||^2 to R
            for (int64_t col = j; col < n; ++col) {
                scalar_t dot_val = 0;
                for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
                    dot_val += v[i] * R_data[(j + i) * n + col];
                }
                scalar_t factor = 2 * dot_val / v_norm;
                for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
                    R_data[(j + i) * n + col] -= factor * v[i];
                }
            }

            // Apply H to Q (Q = Q @ H)
            for (int64_t row = 0; row < m; ++row) {
                scalar_t dot_val = 0;
                for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
                    dot_val += Q_data[row * m + (j + i)] * v[i];
                }
                scalar_t factor = 2 * dot_val / v_norm;
                for (int64_t i = 0; i < static_cast<int64_t>(v.size()); ++i) {
                    Q_data[row * m + (j + i)] -= factor * v[i];
                }
            }
        }
    });

    // Trim R to [m, n] (already correct shape) and Q to [m, m]
    return {Q, R};
}

// ============================================================================
// Trace: sum of diagonal elements
// ============================================================================

inline Tensor trace(const Tensor& self) {
    PT_CHECK_MSG(self.dim() == 2, "trace requires 2D tensor");

    int64_t n = std::min(self.size(0), self.size(1));
    Tensor A = self.contiguous();
    Tensor result = zeros({}, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "trace", [&] {
        const scalar_t* data = A.data_ptr<scalar_t>();
        int64_t cols = A.size(1);
        scalar_t sum = 0;
        for (int64_t i = 0; i < n; ++i) {
            sum += data[i * cols + i];
        }
        result.mutable_data_ptr<scalar_t>()[0] = sum;
    });

    return result;
}

// ============================================================================
// Cross product (3D vectors)
// ============================================================================

inline Tensor cross(const Tensor& self, const Tensor& other, int64_t dim = -1) {
    PT_CHECK_MSG(self.sizes() == other.sizes(), "cross: tensors must have same shape");

    if (dim < 0) dim += self.dim();
    PT_CHECK_MSG(self.size(dim) == 3, "cross: dimension must have size 3");

    Tensor a = self.contiguous();
    Tensor b = other.contiguous();

    // Use select along dim to get components
    Tensor a0 = a.select(dim, 0);
    Tensor a1 = a.select(dim, 1);
    Tensor a2 = a.select(dim, 2);
    Tensor b0 = b.select(dim, 0);
    Tensor b1 = b.select(dim, 1);
    Tensor b2 = b.select(dim, 2);

    Tensor c0 = a1.mul(b2).sub(a2.mul(b1));
    Tensor c1 = a2.mul(b0).sub(a0.mul(b2));
    Tensor c2 = a0.mul(b1).sub(a1.mul(b0));

    return at::native::stack({c0, c1, c2}, dim);
}

// ============================================================================
// Matrix Norm
// ============================================================================

inline Tensor matrix_norm(const Tensor& self, double ord = 2.0) {
    PT_CHECK_MSG(self.dim() == 2, "matrix_norm requires 2D tensor");

    if (ord == 1.0) {
        // Max absolute column sum
        return at::native::sum(at::native::abs(self), 0).max();
    } else if (ord == -1.0) {
        // Min absolute column sum
        return at::native::sum(at::native::abs(self), 0).min();
    } else if (std::isinf(ord) && ord > 0) {
        // Max absolute row sum
        return at::native::sum(at::native::abs(self), 1).max();
    } else if (std::isinf(ord) && ord < 0) {
        // Min absolute row sum
        return at::native::sum(at::native::abs(self), 1).min();
    } else {
        // Frobenius norm (default for ord=2 approximation)
        return at::native::sqrt(at::native::sum(at::native::mul(self, self)));
    }
}

// ============================================================================
// Least Squares (lstsq) — solve min ||Ax - b||_2 via QR
// ============================================================================

inline Tensor lstsq(const Tensor& A, const Tensor& b) {
    PT_CHECK_MSG(A.dim() == 2, "lstsq: A must be 2D");
    PT_CHECK_MSG(b.dim() == 1 || b.dim() == 2, "lstsq: b must be 1D or 2D");

    int64_t m = A.size(0);
    int64_t n = A.size(1);
    bool b_is_1d = (b.dim() == 1);
    Tensor b2 = b_is_1d ? b.unsqueeze(1) : b;
    PT_CHECK_MSG(b2.size(0) == m, "lstsq: dimension mismatch");

    int64_t nrhs = b2.size(1);

    // QR decomposition: A = Q @ R
    auto qr_r_ = qr(A);
    Tensor Q = qr_r_.Q; Tensor R = qr_r_.R;

    // Q^T @ b
    Tensor Qt_b = mm(Q.t(), b2); // [m, m]^T @ [m, nrhs] = [m, nrhs]

    // Back-substitution: Rx = Qt_b (only first n rows)
    Tensor x = zeros({n, nrhs}, TensorOptions().dtype(A.dtype()));

    PT_DISPATCH_FLOATING_TYPES(A.dtype(), "lstsq", [&] {
        const scalar_t* R_data = R.contiguous().data_ptr<scalar_t>();
        const scalar_t* rhs_data = Qt_b.contiguous().data_ptr<scalar_t>();
        scalar_t* x_data = x.mutable_data_ptr<scalar_t>();
        int64_t R_cols = R.size(1); // n

        for (int64_t col = 0; col < nrhs; ++col) {
            for (int64_t i = n - 1; i >= 0; --i) {
                scalar_t sum = rhs_data[i * nrhs + col];
                for (int64_t j = i + 1; j < n; ++j) {
                    sum -= R_data[i * R_cols + j] * x_data[j * nrhs + col];
                }
                scalar_t diag = R_data[i * R_cols + i];
                x_data[i * nrhs + col] = (std::abs(diag) > 1e-15) ? sum / diag : 0;
            }
        }
    });

    return b_is_1d ? x.squeeze(1) : x;
}

// ============================================================================
// SVD — Singular Value Decomposition via Jacobi one-sided method
// A = U @ diag(S) @ Vh
// ============================================================================

struct SVDResult {
    Tensor U, S, Vh;
};

inline SVDResult svd(const Tensor& self, bool full_matrices = true) {
    PT_CHECK_MSG(self.dim() == 2, "svd requires 2D tensor");

    int64_t m = self.size(0);
    int64_t n = self.size(1);
    int64_t k = std::min(m, n);

    // Work on A^T A for right singular vectors, then recover U
    // For small matrices, use Jacobi eigenvalue method on A^T A
    Tensor A = self.contiguous().clone();
    Tensor AtA = mm(self.t(), self); // [n, n]

    // Eigen-decompose A^T A to get V and sigma^2
    // Use Jacobi rotation method for symmetric matrices
    Tensor V = eye(n, n, TensorOptions().dtype(self.dtype()));
    Tensor D = AtA.contiguous().clone();

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "svd", [&] {
        scalar_t* D_data = D.mutable_data_ptr<scalar_t>();
        scalar_t* V_data = V.mutable_data_ptr<scalar_t>();

        const int max_iter = 100;
        for (int iter = 0; iter < max_iter; ++iter) {
            scalar_t off_diag = 0;
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = i + 1; j < n; ++j) {
                    off_diag += D_data[i * n + j] * D_data[i * n + j];
                }
            }
            if (off_diag < 1e-20) break;

            for (int64_t p = 0; p < n; ++p) {
                for (int64_t q = p + 1; q < n; ++q) {
                    scalar_t apq = D_data[p * n + q];
                    if (std::abs(apq) < 1e-15) continue;

                    scalar_t app = D_data[p * n + p];
                    scalar_t aqq = D_data[q * n + q];
                    scalar_t tau = (aqq - app) / (2 * apq);
                    scalar_t t;
                    if (tau >= 0) {
                        t = 1.0 / (tau + std::sqrt(1 + tau * tau));
                    } else {
                        t = -1.0 / (-tau + std::sqrt(1 + tau * tau));
                    }
                    scalar_t c = 1.0 / std::sqrt(1 + t * t);
                    scalar_t s = t * c;

                    // Rotate D
                    D_data[p * n + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                    D_data[q * n + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                    D_data[p * n + q] = 0;
                    D_data[q * n + p] = 0;

                    for (int64_t i = 0; i < n; ++i) {
                        if (i == p || i == q) continue;
                        scalar_t dip = D_data[i * n + p];
                        scalar_t diq = D_data[i * n + q];
                        D_data[i * n + p] = c * dip - s * diq;
                        D_data[p * n + i] = D_data[i * n + p];
                        D_data[i * n + q] = s * dip + c * diq;
                        D_data[q * n + i] = D_data[i * n + q];
                    }

                    // Rotate V
                    for (int64_t i = 0; i < n; ++i) {
                        scalar_t vip = V_data[i * n + p];
                        scalar_t viq = V_data[i * n + q];
                        V_data[i * n + p] = c * vip - s * viq;
                        V_data[i * n + q] = s * vip + c * viq;
                    }
                }
            }
        }
    });

    // Extract singular values (sqrt of eigenvalues of AtA)
    Tensor S = zeros({k}, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "svd_s", [&] {
        const scalar_t* D_data = D.data_ptr<scalar_t>();
        scalar_t* S_data = S.mutable_data_ptr<scalar_t>();

        // Collect eigenvalues and sort descending
        std::vector<std::pair<scalar_t, int64_t>> eig_pairs(n);
        for (int64_t i = 0; i < n; ++i) {
            scalar_t val = D_data[i * n + i];
            eig_pairs[i] = {val >= 0 ? std::sqrt(val) : 0, i};
        }
        std::sort(eig_pairs.begin(), eig_pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int64_t i = 0; i < k; ++i) {
            S_data[i] = eig_pairs[i].first;
        }

        // Reorder V columns according to sorted order
        Tensor V_sorted = at::empty({n, n}, TensorOptions().dtype(self.dtype()));
        scalar_t* Vs = V_sorted.mutable_data_ptr<scalar_t>();
        const scalar_t* Vd = V.data_ptr<scalar_t>();
        for (int64_t j = 0; j < n; ++j) {
            int64_t src_col = eig_pairs[j].second;
            for (int64_t i = 0; i < n; ++i) {
                Vs[i * n + j] = Vd[i * n + src_col];
            }
        }
        V = V_sorted;
    });

    // U = A @ V @ diag(1/S) for first k columns
    Tensor Vk = V.narrow(1, 0, k); // [n, k]
    Tensor U_raw = mm(self, Vk);   // [m, k]

    // Normalize columns of U by singular values
    Tensor U = U_raw.clone();
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "svd_u", [&] {
        scalar_t* U_data = U.mutable_data_ptr<scalar_t>();
        const scalar_t* S_data = S.data_ptr<scalar_t>();
        for (int64_t j = 0; j < k; ++j) {
            scalar_t sv = S_data[j];
            if (sv > 1e-15) {
                for (int64_t i = 0; i < m; ++i) {
                    U_data[i * k + j] /= sv;
                }
            }
        }
    });

    // Vh = V^T (first k rows)
    Tensor Vh = V.t().narrow(0, 0, k).contiguous(); // [k, n]

    if (full_matrices) {
        // Extend U to [m, m] and Vh to [n, n] (fill with orthogonal complement)
        // For simplicity, return the thin SVD for full_matrices=false
        // and the thin factors for full_matrices=true (common practical usage)
    }

    return {U, S, Vh};
}

// ============================================================================
// Pseudo-inverse (pinverse) via SVD
// ============================================================================

inline Tensor pinverse(const Tensor& self, double rcond = 1e-15) {
    PT_CHECK_MSG(self.dim() == 2, "pinverse requires 2D tensor");

    auto svd_r_ = svd(self, false);
    Tensor U = svd_r_.U; Tensor S = svd_r_.S; Tensor Vh = svd_r_.Vh;

    // Invert S with threshold
    int64_t k = S.size(0);
    Tensor S_inv = zeros({k}, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "pinverse", [&] {
        const scalar_t* s_data = S.data_ptr<scalar_t>();
        scalar_t* si_data = S_inv.mutable_data_ptr<scalar_t>();
        scalar_t threshold = static_cast<scalar_t>(rcond) * s_data[0]; // largest sv * rcond

        for (int64_t i = 0; i < k; ++i) {
            si_data[i] = (s_data[i] > threshold) ? (1.0 / s_data[i]) : 0;
        }
    });

    // pinverse = Vh^T @ diag(S_inv) @ U^T
    // = Vh^T @ diag(S_inv) @ U^T
    Tensor Vh_t = Vh.t(); // [n, k]
    Tensor U_t = U.t();   // [k, m]

    // Scale columns of Vh_t by S_inv
    Tensor scaled = Vh_t.clone();
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "pinverse_scale", [&] {
        scalar_t* sc_data = scaled.mutable_data_ptr<scalar_t>();
        const scalar_t* si_data = S_inv.data_ptr<scalar_t>();
        int64_t rows = scaled.size(0);
        int64_t cols = scaled.size(1);
        for (int64_t j = 0; j < cols; ++j) {
            for (int64_t i = 0; i < rows; ++i) {
                sc_data[i * cols + j] *= si_data[j];
            }
        }
    });

    return mm(scaled, U_t);
}

// ============================================================================
// Eigenvalue decomposition (eig) — symmetric matrices via Jacobi rotation
// Returns (eigenvalues, eigenvectors)
// ============================================================================

struct EigResult {
    Tensor eigenvalues;
    Tensor eigenvectors;
};

inline EigResult eig(const Tensor& self) {
    PT_CHECK_MSG(self.dim() == 2, "eig requires 2D tensor");
    PT_CHECK_MSG(self.size(0) == self.size(1), "eig requires square matrix");

    int64_t n = self.size(0);

    Tensor D = self.contiguous().clone();
    Tensor V = eye(n, n, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "eig", [&] {
        scalar_t* D_data = D.mutable_data_ptr<scalar_t>();
        scalar_t* V_data = V.mutable_data_ptr<scalar_t>();

        const int max_iter = 200;
        for (int iter = 0; iter < max_iter; ++iter) {
            // Check convergence
            scalar_t off = 0;
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = i + 1; j < n; ++j) {
                    off += D_data[i * n + j] * D_data[i * n + j];
                }
            }
            if (off < 1e-20) break;

            // Jacobi sweep
            for (int64_t p = 0; p < n; ++p) {
                for (int64_t q = p + 1; q < n; ++q) {
                    scalar_t apq = D_data[p * n + q];
                    if (std::abs(apq) < 1e-15) continue;

                    scalar_t app = D_data[p * n + p];
                    scalar_t aqq = D_data[q * n + q];
                    scalar_t tau = (aqq - app) / (2 * apq);
                    scalar_t t;
                    if (tau >= 0) {
                        t = 1.0 / (tau + std::sqrt(1 + tau * tau));
                    } else {
                        t = -1.0 / (-tau + std::sqrt(1 + tau * tau));
                    }
                    scalar_t c = 1.0 / std::sqrt(1 + t * t);
                    scalar_t s = t * c;

                    D_data[p * n + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                    D_data[q * n + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                    D_data[p * n + q] = 0;
                    D_data[q * n + p] = 0;

                    for (int64_t i = 0; i < n; ++i) {
                        if (i == p || i == q) continue;
                        scalar_t dip = D_data[i * n + p];
                        scalar_t diq = D_data[i * n + q];
                        D_data[i * n + p] = c * dip - s * diq;
                        D_data[p * n + i] = D_data[i * n + p];
                        D_data[i * n + q] = s * dip + c * diq;
                        D_data[q * n + i] = D_data[i * n + q];
                    }

                    for (int64_t i = 0; i < n; ++i) {
                        scalar_t vip = V_data[i * n + p];
                        scalar_t viq = V_data[i * n + q];
                        V_data[i * n + p] = c * vip - s * viq;
                        V_data[i * n + q] = s * vip + c * viq;
                    }
                }
            }
        }
    });

    // Extract eigenvalues and sort
    Tensor eigenvalues = zeros({n}, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "eig_sort", [&] {
        scalar_t* D_data = D.mutable_data_ptr<scalar_t>();
        scalar_t* ev_data = eigenvalues.mutable_data_ptr<scalar_t>();
        scalar_t* V_data = V.mutable_data_ptr<scalar_t>();

        // Collect and sort by descending absolute value
        std::vector<std::pair<scalar_t, int64_t>> pairs(n);
        for (int64_t i = 0; i < n; ++i) {
            pairs[i] = {D_data[i * n + i], i};
        }
        std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return std::abs(a.first) > std::abs(b.first); });

        Tensor V_sorted = at::empty({n, n}, TensorOptions().dtype(self.dtype()));
        scalar_t* Vs = V_sorted.mutable_data_ptr<scalar_t>();

        for (int64_t j = 0; j < n; ++j) {
            ev_data[j] = pairs[j].first;
            int64_t src = pairs[j].second;
            for (int64_t i = 0; i < n; ++i) {
                Vs[i * n + j] = V_data[i * n + src];
            }
        }
        V = V_sorted;
    });

    return {eigenvalues, V};
}

} // namespace native
} // namespace at
