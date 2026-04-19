#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include "aten/src/ATen/native/cpu/ReduceOps.h"
#include "aten/src/ATen/native/cpu/MathOps.h"
#include "aten/src/ATen/native/cpu/ShapeOps.h"
#include "aten/src/ATen/native/cpu/PromeBLAS.h"
#include "aten/src/ATen/native/cpu/hot_loops.h"
#ifdef PT_USE_NMQUAD
#include "aten/src/ATen/nmquad/NMQuadOps.h"
#endif
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
#ifdef PT_USE_NMQUAD
    // NM QUAD dispatch: if either tensor is on nmquad device
    if (self.is_nmquad() || other.is_nmquad()) {
        return at::nmquad::matmul_nmquad(self.contiguous(), other.contiguous(), 0);
    }
#endif

    // FAST PATH: both trusted (float32, contiguous, CPU) — skip all checks
    if (self.is_trusted() && other.is_trusted()) {
        int64_t M = self.size(0), K = self.size(1), N = other.size(1);
        Tensor result = at::empty({M, N});
        hot::sgemm(M, K, N, 1.0f, self.data_ptr<float>(), K,
                   other.data_ptr<float>(), N, 0.0f,
                   result.mutable_data_ptr<float>(), N);
        return result;
    }

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
            hot::sgemm_nt(M, K, N, 1.0f, A_data, K, B_data, ldb, 0.0f, C, N);
        } else {
            Tensor B = other.contiguous();
            const float* B_data = B.data_ptr<float>();
            hot::sgemm(M, K, N, 1.0f, A_data, K, B_data, N, 0.0f, C, N);
        }

        return result;
    }

    // Non-float fallback: make contiguous and use scalar loop
    Tensor A = self.contiguous();
    Tensor B = other.contiguous();
    Tensor result = zeros({M, N}, TensorOptions().dtype(result_dtype).device(A.device()));

    if (c10::isComplexType(result_dtype)) {
        // Naive O(M*N*K) complex matmul. No SIMD initially.
        PT_DISPATCH_COMPLEX_TYPES(result_dtype, "mm_complex", [&] {
            const scalar_t* A_data = A.data_ptr<scalar_t>();
            const scalar_t* B_data = B.data_ptr<scalar_t>();
            scalar_t* C = result.mutable_data_ptr<scalar_t>();
            for (int64_t i = 0; i < M; ++i) {
                for (int64_t j = 0; j < N; ++j) {
                    scalar_t sum(0);
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A_data[i * K + k] * B_data[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        });
        return result;
    }

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
    // FAST PATH: trusted tensors skip all checks and contiguous()
    if (self.is_trusted() && vec.is_trusted()) {
        int64_t M = self.size(0), N = self.size(1);
        Tensor result = at::zeros({M});
        hot::sgemv(M, N, 1.0f, self.data_ptr<float>(), N, vec.data_ptr<float>(),
                   0.0f, result.mutable_data_ptr<float>());
        return result;
    }

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
        hot::sgemv(M, N, 1.0f, A.data_ptr<float>(), N, x.data_ptr<float>(),
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
            hot::sgemm(M, K, N, 1.0f,
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
        float val = hot::sdot(n, a.data_ptr<float>(), b.data_ptr<float>());
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
            hot::sgemm_nt(M, K, N, alpha_val, A_data, K,
                          mat2.data_ptr<float>(), mat2.stride(1),
                          beta_val, C, N);
        } else {
            Tensor B = mat2.contiguous();
            hot::sgemm(M, K, N, alpha_val, A_data, K,
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

// Forward declaration of the full-featured einsum implementation
// (defined in Einsum.h, which must be included after LinearAlgebra.h).
inline Tensor einsum_impl(const std::string& equation, const std::vector<Tensor>& operands);

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

    // General fallback: dispatch to the full einsum implementation
    // (handles 3+ operands, ellipsis, repeated labels, etc.).
    return einsum_impl(equation, tensors);
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
        // Extend U [m,k] -> [m,m] and Vh [k,n] -> [n,n] via Gram-Schmidt on
        // standard basis vectors, orthogonalizing against existing columns/rows.
        PT_DISPATCH_FLOATING_TYPES(self.dtype(), "svd_full", [&] {
            auto extend_columns = [](Tensor& Q, int64_t rows, int64_t cur_cols, int64_t target_cols) {
                if (cur_cols >= target_cols) return;
                Tensor Q_full = empty({rows, target_cols},
                                      TensorOptions().dtype(Q.dtype()).device(Q.device()));
                scalar_t* qf = Q_full.mutable_data_ptr<scalar_t>();
                const scalar_t* q = Q.data_ptr<scalar_t>();
                // Copy existing columns.
                for (int64_t i = 0; i < rows; ++i)
                    for (int64_t j = 0; j < cur_cols; ++j)
                        qf[i * target_cols + j] = q[i * cur_cols + j];
                // Fill additional columns via Gram-Schmidt on successive e_basis vectors.
                int64_t basis_idx = 0;
                for (int64_t new_col = cur_cols; new_col < target_cols; ++new_col) {
                    bool placed = false;
                    while (!placed && basis_idx < rows) {
                        // v = e_{basis_idx}
                        std::vector<scalar_t> v(rows, scalar_t(0));
                        v[basis_idx] = scalar_t(1);
                        // Orthogonalize against all previous columns.
                        for (int64_t c = 0; c < new_col; ++c) {
                            scalar_t dot = scalar_t(0);
                            for (int64_t i = 0; i < rows; ++i)
                                dot += qf[i * target_cols + c] * v[i];
                            for (int64_t i = 0; i < rows; ++i)
                                v[i] -= dot * qf[i * target_cols + c];
                        }
                        // Norm check.
                        scalar_t norm2 = scalar_t(0);
                        for (int64_t i = 0; i < rows; ++i) norm2 += v[i] * v[i];
                        basis_idx++;
                        if (norm2 > scalar_t(1e-20)) {
                            scalar_t inv = scalar_t(1) / std::sqrt(norm2);
                            for (int64_t i = 0; i < rows; ++i)
                                qf[i * target_cols + new_col] = v[i] * inv;
                            placed = true;
                        }
                    }
                    PT_CHECK_MSG(placed, "svd: failed to extend to full_matrices (degenerate)");
                }
                Q = Q_full;
            };
            // U: extend from [m,k] to [m,m]
            extend_columns(U, m, k, m);
            // Vh rows correspond to rows of V^T. V is [n,n] thin -> Vh [k,n]. Extend to [n,n].
            // Work on V^T shape: treat Vh as [target_rows=n, cols=n] by transposing logic.
            // Easier: extend V (currently [n,k]) to [n,n], then Vh_full = V_full^T.
            Tensor V_full = V.narrow(1, 0, k).contiguous();
            extend_columns(V_full, n, k, n);
            Vh = V_full.t().contiguous();
        });
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

// ============================================================================
// Randomized SVD (Halko-Martinsson-Tropp 2011)
// Computes top-k singular values/vectors in O(mn*k) instead of O(mn^2).
// Reference: "Finding Structure with Randomness" (2011)
// ============================================================================

inline SVDResult randomized_svd(const Tensor& self, int64_t k,
                                int64_t oversampling = 10, int64_t n_iter = 2) {
    PT_CHECK_MSG(self.dim() == 2, "randomized_svd requires 2D tensor");

    int64_t m = self.size(0);
    int64_t n = self.size(1);
    int64_t min_mn = std::min(m, n);
    PT_CHECK_MSG(k > 0 && k <= min_mn,
        "randomized_svd: k must be in [1, min(m,n)], got k=", k,
        " for matrix [", m, ", ", n, "]");

    // Clamp oversampling so total sketch size does not exceed n
    int64_t p = std::min(oversampling, min_mn - k);
    if (p < 0) p = 0;
    int64_t l = k + p;  // sketch size

    Tensor A = self.contiguous();

    // Step 1: Generate random Gaussian matrix Omega [n, l]
    Tensor Omega = at::randn({n, l}, TensorOptions().dtype(self.dtype()));

    // Step 2: Form Y = A @ Omega  [m, l]
    Tensor Y = mm(A, Omega);

    // Step 3: Power iteration to improve accuracy for slowly decaying spectra
    // Each iteration: Y = A @ (A^T @ Y)
    Tensor At = A.t();  // [n, m]
    for (int64_t iter = 0; iter < n_iter; ++iter) {
        // Orthogonalize Y for numerical stability before each power step
        auto qr_y = qr(Y);
        Y = qr_y.Q.narrow(1, 0, l);  // [m, l] — thin Q

        Tensor Z = mm(At, Y);    // [n, l]
        // Orthogonalize Z too
        auto qr_z = qr(Z);
        Z = qr_z.Q.narrow(1, 0, l);  // [n, l]

        Y = mm(A, Z);             // [m, l]
    }

    // Step 4: QR decomposition of Y to get orthonormal basis Q
    auto qr_result = qr(Y);
    Tensor Q = qr_result.Q.narrow(1, 0, l);  // [m, l]

    // Step 5: Form small matrix B = Q^T @ A  [l, n]
    Tensor B = mm(Q.t(), A);

    // Step 6: Full SVD of small B (l x n, where l << m typically)
    auto svd_b = svd(B, false);
    Tensor Ub = svd_b.U;   // [l, min(l,n)]
    Tensor S  = svd_b.S;   // [min(l,n)]
    Tensor Vt = svd_b.Vh;  // [min(l,n), n]

    // Step 7: Recover U = Q @ Ub  [m, min(l,n)]
    Tensor U = mm(Q, Ub);

    // Truncate to top-k
    Tensor U_k  = U.narrow(1, 0, k).contiguous();   // [m, k]
    Tensor S_k  = S.narrow(0, 0, k).contiguous();    // [k]
    Tensor Vt_k = Vt.narrow(0, 0, k).contiguous();   // [k, n]

    return {U_k, S_k, Vt_k};
}

// ============================================================================
// Truncated SVD via Lanczos bidiagonalization
// Computes top-k singular values/vectors in O(mn*k*iterations).
// Uses Lanczos iteration on A^T A to find dominant eigenvalues.
// ============================================================================

inline SVDResult lanczos_svd(const Tensor& self, int64_t k,
                             int64_t max_iter = 100, double tol = 1e-6) {
    PT_CHECK_MSG(self.dim() == 2, "lanczos_svd requires 2D tensor");

    int64_t m = self.size(0);
    int64_t n = self.size(1);
    int64_t min_mn = std::min(m, n);
    PT_CHECK_MSG(k > 0 && k <= min_mn,
        "lanczos_svd: k must be in [1, min(m,n)], got k=", k,
        " for matrix [", m, ", ", n, "]");

    Tensor A = self.contiguous();

    // Use Lanczos bidiagonalization
    // Build a tridiagonal matrix T such that eigenvalues of T
    // approximate eigenvalues of A^T A (= squared singular values of A).
    //
    // We keep Lanczos vectors for later recovery of singular vectors.

    // Number of Lanczos steps: at least k, but run more for accuracy
    int64_t lanczos_steps = std::min(static_cast<int64_t>(max_iter),
                                     std::min(n, std::max(k + 20, 2 * k)));

    // Storage for Lanczos vectors V [lanczos_steps, n] (rows are v_j)
    // and tridiagonal coefficients alpha[j], beta[j]
    Tensor V_storage = at::zeros({lanczos_steps, n}, TensorOptions().dtype(self.dtype()));
    std::vector<double> alpha_vec(lanczos_steps, 0.0);
    std::vector<double> beta_vec(lanczos_steps, 0.0);

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "lanczos_svd", [&] {
        scalar_t* V_data = V_storage.mutable_data_ptr<scalar_t>();

        // Step 1: Start with random unit vector v0
        Tensor v = at::randn({n}, TensorOptions().dtype(self.dtype()));
        {
            // Normalize
            scalar_t* v_ptr = v.mutable_data_ptr<scalar_t>();
            scalar_t nrm = 0;
            for (int64_t i = 0; i < n; ++i) nrm += v_ptr[i] * v_ptr[i];
            nrm = std::sqrt(nrm);
            if (nrm > 1e-15) {
                for (int64_t i = 0; i < n; ++i) v_ptr[i] /= nrm;
            }
        }

        // Store v0
        {
            const scalar_t* v_ptr = v.data_ptr<scalar_t>();
            for (int64_t i = 0; i < n; ++i) V_data[0 * n + i] = v_ptr[i];
        }

        Tensor v_prev = at::zeros({n}, TensorOptions().dtype(self.dtype()));
        scalar_t beta_prev = 0;

        int64_t actual_steps = lanczos_steps;

        for (int64_t j = 0; j < lanczos_steps; ++j) {
            // w = A^T @ (A @ v) - beta_{j-1} * v_{j-1}
            // This computes (A^T A) v without forming A^T A explicitly
            Tensor Av = mm(A, v.unsqueeze(1));  // [m, 1]
            Tensor AtAv = mm(A.t(), Av);         // [n, 1]
            Tensor w = AtAv.squeeze(1);           // [n]

            if (j > 0) {
                // w = w - beta_{j-1} * v_{j-1}
                scalar_t* w_ptr = w.mutable_data_ptr<scalar_t>();
                const scalar_t* vp_ptr = v_prev.data_ptr<scalar_t>();
                for (int64_t i = 0; i < n; ++i) {
                    w_ptr[i] -= beta_prev * vp_ptr[i];
                }
            }

            // alpha_j = dot(w, v)
            scalar_t alpha_j = 0;
            {
                const scalar_t* w_ptr = w.data_ptr<scalar_t>();
                const scalar_t* v_ptr = v.data_ptr<scalar_t>();
                for (int64_t i = 0; i < n; ++i) alpha_j += w_ptr[i] * v_ptr[i];
            }
            alpha_vec[j] = static_cast<double>(alpha_j);

            // w = w - alpha_j * v
            {
                scalar_t* w_ptr = w.mutable_data_ptr<scalar_t>();
                const scalar_t* v_ptr = v.data_ptr<scalar_t>();
                for (int64_t i = 0; i < n; ++i) w_ptr[i] -= alpha_j * v_ptr[i];
            }

            // Full re-orthogonalization against all previous Lanczos vectors
            // (critical for numerical stability)
            for (int64_t prev = 0; prev <= j; ++prev) {
                scalar_t dot_val = 0;
                scalar_t* w_ptr = w.mutable_data_ptr<scalar_t>();
                for (int64_t i = 0; i < n; ++i) {
                    dot_val += w_ptr[i] * V_data[prev * n + i];
                }
                for (int64_t i = 0; i < n; ++i) {
                    w_ptr[i] -= dot_val * V_data[prev * n + i];
                }
            }

            // beta_j = ||w||
            scalar_t beta_j = 0;
            {
                const scalar_t* w_ptr = w.data_ptr<scalar_t>();
                for (int64_t i = 0; i < n; ++i) beta_j += w_ptr[i] * w_ptr[i];
            }
            beta_j = std::sqrt(beta_j);
            beta_vec[j] = static_cast<double>(beta_j);

            // Convergence check: if beta is tiny, invariant subspace found
            if (beta_j < static_cast<scalar_t>(tol)) {
                actual_steps = j + 1;
                break;
            }

            // v_prev = v, v = w / beta_j
            v_prev = v.clone();
            beta_prev = beta_j;
            v = w.div(at::full({}, beta_j, TensorOptions().dtype(self.dtype())));

            // Store v_{j+1}
            if (j + 1 < lanczos_steps) {
                const scalar_t* v_ptr = v.data_ptr<scalar_t>();
                for (int64_t i = 0; i < n; ++i) V_data[(j + 1) * n + i] = v_ptr[i];
            }
        }

        lanczos_steps = actual_steps;
    });

    // Step 2: Build tridiagonal matrix T [lanczos_steps, lanczos_steps]
    Tensor T = at::zeros({lanczos_steps, lanczos_steps}, TensorOptions().dtype(self.dtype()));
    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "lanczos_tridiag", [&] {
        scalar_t* T_data = T.mutable_data_ptr<scalar_t>();
        for (int64_t j = 0; j < lanczos_steps; ++j) {
            T_data[j * lanczos_steps + j] = static_cast<scalar_t>(alpha_vec[j]);
            if (j + 1 < lanczos_steps) {
                T_data[j * lanczos_steps + (j + 1)] = static_cast<scalar_t>(beta_vec[j]);
                T_data[(j + 1) * lanczos_steps + j] = static_cast<scalar_t>(beta_vec[j]);
            }
        }
    });

    // Step 3: Eigendecompose T (small matrix — Jacobi is fine here)
    auto eig_result = eig(T);
    Tensor eigenvalues = eig_result.eigenvalues;   // [lanczos_steps], sorted desc by |val|
    Tensor eigenvectors = eig_result.eigenvectors;  // [lanczos_steps, lanczos_steps]

    // Eigenvalues of T approximate eigenvalues of A^T A = sigma^2
    // They are already sorted by descending absolute value from eig()

    // Step 4: Extract top-k singular values
    int64_t k_actual = std::min(k, lanczos_steps);
    Tensor S = at::zeros({k_actual}, TensorOptions().dtype(self.dtype()));

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "lanczos_sigma", [&] {
        const scalar_t* ev_data = eigenvalues.data_ptr<scalar_t>();
        scalar_t* s_data = S.mutable_data_ptr<scalar_t>();
        for (int64_t i = 0; i < k_actual; ++i) {
            scalar_t val = ev_data[i];
            s_data[i] = (val >= 0) ? std::sqrt(val) : 0;
        }
    });

    // Step 5: Recover right singular vectors V = V_lanczos^T @ eigvecs_of_T
    // V_lanczos is [lanczos_steps, n], eigvecs is [lanczos_steps, lanczos_steps]
    // Right singular vectors: V_svd = V_lanczos^T @ eigvecs[:, :k]  → [n, k]
    Tensor V_lanczos = V_storage.narrow(0, 0, lanczos_steps);  // [lanczos_steps, n]
    Tensor eig_k = eigenvectors.narrow(1, 0, k_actual);        // [lanczos_steps, k]

    // V_svd = V_lanczos^T @ eig_k = [n, lanczos_steps] @ [lanczos_steps, k] = [n, k]
    Tensor V_svd = mm(V_lanczos.t(), eig_k);  // [n, k]

    // Vh = V_svd^T  [k, n]
    Tensor Vh = V_svd.t().contiguous();

    // Step 6: Recover left singular vectors U = A @ V_svd @ diag(1/S)
    Tensor U = mm(self, V_svd);  // [m, k]

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "lanczos_U", [&] {
        scalar_t* U_data = U.mutable_data_ptr<scalar_t>();
        const scalar_t* s_data = S.data_ptr<scalar_t>();
        for (int64_t j = 0; j < k_actual; ++j) {
            scalar_t sv = s_data[j];
            if (sv > static_cast<scalar_t>(1e-15)) {
                for (int64_t i = 0; i < m; ++i) {
                    U_data[i * k_actual + j] /= sv;
                }
            }
        }
    });

    return {U.contiguous(), S, Vh};
}

// ============================================================================
// Fast SVD dispatch — automatically selects the best algorithm
//   method="auto"       : randomized if k < min(m,n)/4, else full Jacobi
//   method="randomized" : Halko-Martinsson-Tropp randomized SVD
//   method="lanczos"    : Lanczos bidiagonalization
//   method="full"       : existing Jacobi rotation (O(n^3))
// ============================================================================

inline SVDResult svd_fast(const Tensor& self, int64_t k = -1,
                          const std::string& method = "auto") {
    PT_CHECK_MSG(self.dim() == 2, "svd_fast requires 2D tensor");

    int64_t m = self.size(0);
    int64_t n = self.size(1);
    int64_t min_mn = std::min(m, n);

    if (k < 0) k = min_mn;  // full SVD
    PT_CHECK_MSG(k <= min_mn,
        "svd_fast: k=", k, " exceeds min(m,n)=", min_mn);

    if (method == "randomized" || (method == "auto" && k < min_mn / 4)) {
        return randomized_svd(self, k);
    } else if (method == "lanczos") {
        return lanczos_svd(self, k);
    } else {
        // Full Jacobi SVD, then truncate to k if needed
        auto result = svd(self, false);
        if (k < min_mn) {
            return {
                result.U.narrow(1, 0, k).contiguous(),
                result.S.narrow(0, 0, k).contiguous(),
                result.Vh.narrow(0, 0, k).contiguous()
            };
        }
        return result;
    }
}

// ============================================================================
// Low-rank approximation: A ≈ U @ diag(S) @ Vt  (best rank-k approx)
// Uses randomized SVD for O(mn*k) complexity.
// ============================================================================

inline Tensor low_rank_approx(const Tensor& self, int64_t k) {
    PT_CHECK_MSG(self.dim() == 2, "low_rank_approx requires 2D tensor");
    PT_CHECK_MSG(k > 0, "low_rank_approx: k must be positive, got ", k);
    PT_CHECK_MSG(k <= std::min(self.size(0), self.size(1)),
        "low_rank_approx: k=", k, " exceeds min(m,n)=",
        std::min(self.size(0), self.size(1)));

    auto result = randomized_svd(self, k);

    // Reconstruct: U * diag(S) * Vt
    // Scale columns of U by corresponding singular values
    Tensor US = result.U.contiguous().clone();
    int64_t m_rows = US.size(0);

    PT_DISPATCH_FLOATING_TYPES(self.dtype(), "low_rank_approx", [&] {
        scalar_t* us_data = US.mutable_data_ptr<scalar_t>();
        const scalar_t* s_data = result.S.data_ptr<scalar_t>();
        for (int64_t j = 0; j < k; ++j) {
            scalar_t s = s_data[j];
            for (int64_t i = 0; i < m_rows; ++i) {
                us_data[i * k + j] *= s;
            }
        }
    });

    return mm(US, result.Vh);  // [m, k] @ [k, n] = [m, n]
}

// ============================================================================
// Weight compression via SVD — replace W[m,n] with {A[m,r], B[r,n]}
// Forward: input[batch, n] @ B^T @ A^T — two smaller matmuls
// When rank << min(m,n), inference is significantly faster.
// ============================================================================

struct CompressedWeight {
    Tensor A;  // [m, rank]
    Tensor B;  // [rank, n]

    // Apply compressed weight: input[batch, n] -> output[batch, m]
    // Equivalent to input @ W^T where W ≈ A @ B
    Tensor forward(const Tensor& input) const {
        PT_CHECK_MSG(input.dim() == 2, "CompressedWeight::forward requires 2D input");
        // input @ B^T @ A^T = input @ (A @ B)^T
        Tensor tmp = mm(input, B.t());  // [batch, rank]
        return mm(tmp, A.t());          // [batch, m]
    }

    // Reconstruct the approximate weight matrix W ≈ A @ B
    Tensor reconstruct() const {
        return mm(A, B);  // [m, n]
    }
};

inline CompressedWeight compress_weight(const Tensor& W, int64_t rank) {
    PT_CHECK_MSG(W.dim() == 2, "compress_weight requires 2D weight tensor");
    PT_CHECK_MSG(rank > 0, "compress_weight: rank must be positive, got ", rank);
    PT_CHECK_MSG(rank <= std::min(W.size(0), W.size(1)),
        "compress_weight: rank=", rank, " exceeds min(m,n)=",
        std::min(W.size(0), W.size(1)));

    auto svd_r = randomized_svd(W, rank);
    // W ≈ U @ diag(S) @ Vt
    // Split as: A = U @ diag(sqrt(S)),  B = diag(sqrt(S)) @ Vt
    // So A @ B = U @ diag(S) @ Vt ≈ W

    int64_t m = W.size(0);
    int64_t n = W.size(1);

    Tensor A_out = svd_r.U.contiguous().clone();   // [m, rank]
    Tensor B_out = svd_r.Vh.contiguous().clone();  // [rank, n]

    PT_DISPATCH_FLOATING_TYPES(W.dtype(), "compress_weight", [&] {
        const scalar_t* s_data = svd_r.S.data_ptr<scalar_t>();
        scalar_t* a_data = A_out.mutable_data_ptr<scalar_t>();
        scalar_t* b_data = B_out.mutable_data_ptr<scalar_t>();

        for (int64_t r = 0; r < rank; ++r) {
            scalar_t sqrt_s = std::sqrt(s_data[r]);

            // Scale column r of A by sqrt(S[r])
            for (int64_t i = 0; i < m; ++i) {
                a_data[i * rank + r] *= sqrt_s;
            }
            // Scale row r of B by sqrt(S[r])
            for (int64_t j = 0; j < n; ++j) {
                b_data[r * n + j] *= sqrt_s;
            }
        }
    });

    return {A_out, B_out};
}

} // namespace native
} // namespace at

// Pull in the full einsum implementation AFTER LinearAlgebra.h defines bmm /
// permute etc. Forward-declared `einsum_impl` above resolves to this inline
// definition. Included here (not at the top) to avoid a circular dependency:
// Einsum.h itself needs bmm/mm/permute from this file.
#include "aten/src/ATen/native/cpu/Einsum.h"
