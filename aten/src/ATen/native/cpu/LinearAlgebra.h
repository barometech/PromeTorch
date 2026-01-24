#pragma once

#include "aten/src/ATen/core/Tensor.h"
#include "aten/src/ATen/core/TensorFactory.h"
#include <cmath>

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
    Tensor result = zeros({M, N}, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_FLOATING_TYPES(result_dtype, "mm", [&] {
        const scalar_t* A = self.data_ptr<scalar_t>();
        const scalar_t* B = other.data_ptr<scalar_t>();
        scalar_t* C = result.mutable_data_ptr<scalar_t>();

        // Basic matrix multiplication with OpenMP parallelization
        // For production, should use BLAS (OpenBLAS/MKL)
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                scalar_t sum = 0;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
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

    int64_t M = self.size(0);
    int64_t N = self.size(1);

    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), vec.dtype());
    Tensor result = zeros({M}, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_FLOATING_TYPES(result_dtype, "mv", [&] {
        const scalar_t* A = self.data_ptr<scalar_t>();
        const scalar_t* x = vec.data_ptr<scalar_t>();
        scalar_t* y = result.mutable_data_ptr<scalar_t>();

        for (int64_t i = 0; i < M; ++i) {
            scalar_t sum = 0;
            for (int64_t j = 0; j < N; ++j) {
                sum += A[i * N + j] * x[j];
            }
            y[i] = sum;
        }
    });

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

    int64_t batch = self.size(0);
    int64_t M = self.size(1);
    int64_t K = self.size(2);
    int64_t N = other.size(2);

    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());
    Tensor result = zeros({batch, M, N}, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_FLOATING_TYPES(result_dtype, "bmm", [&] {
        const scalar_t* A = self.data_ptr<scalar_t>();
        const scalar_t* B = other.data_ptr<scalar_t>();
        scalar_t* C = result.mutable_data_ptr<scalar_t>();

        for (int64_t b = 0; b < batch; ++b) {
            const scalar_t* A_b = A + b * M * K;
            const scalar_t* B_b = B + b * K * N;
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

    int64_t n = self.size(0);
    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());
    Tensor result = zeros({}, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_FLOATING_TYPES(result_dtype, "dot", [&] {
        const scalar_t* a = self.data_ptr<scalar_t>();
        const scalar_t* b = other.data_ptr<scalar_t>();
        scalar_t sum = 0;

        for (int64_t i = 0; i < n; ++i) {
            sum += a[i] * b[i];
        }

        result.mutable_data_ptr<scalar_t>()[0] = sum;
    });

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

    int64_t M = self.size(0);
    int64_t N = other.size(0);

    c10::ScalarType result_dtype = c10::promoteTypes(self.dtype(), other.dtype());
    Tensor result = empty({M, N}, TensorOptions().dtype(result_dtype).device(self.device()));

    PT_DISPATCH_ALL_TYPES(result_dtype, "outer", [&] {
        const scalar_t* a = self.data_ptr<scalar_t>();
        const scalar_t* b = other.data_ptr<scalar_t>();
        scalar_t* out = result.mutable_data_ptr<scalar_t>();

        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                out[i * N + j] = a[i] * b[j];
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

    PT_DISPATCH_FLOATING_TYPES(result.dtype(), "addmm", [&] {
        scalar_t beta_val = beta.to<scalar_t>();
        scalar_t alpha_val = alpha.to<scalar_t>();

        const scalar_t* A = mat1.data_ptr<scalar_t>();
        const scalar_t* B = mat2.data_ptr<scalar_t>();
        scalar_t* C = result.mutable_data_ptr<scalar_t>();

        // Scale existing values by beta
        int64_t total = M * N;
        for (int64_t i = 0; i < total; ++i) {
            C[i] *= beta_val;
        }

        // Add alpha * A @ B
        for (int64_t i = 0; i < M; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                scalar_t sum = 0;
                for (int64_t k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] += alpha_val * sum;
            }
        }
    });

    return result;
}

} // namespace native
} // namespace at
