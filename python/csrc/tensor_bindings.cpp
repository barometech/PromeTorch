// ============================================================================
// PromeTorch Python Bindings - Tensor
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <sstream>
#include <limits>
#include <cstring>
#include <tuple>

#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"

namespace py = pybind11;

// ============================================================================
// Helper: Convert numpy array to Tensor
// ============================================================================

at::Tensor numpy_to_tensor(py::array arr, bool requires_grad = false) {
    // Get buffer info
    py::buffer_info buf = arr.request();

    // Determine dtype
    c10::ScalarType dtype;
    if (buf.format == py::format_descriptor<float>::format()) {
        dtype = c10::ScalarType::Float;
    } else if (buf.format == py::format_descriptor<double>::format()) {
        dtype = c10::ScalarType::Double;
    } else if (buf.format == py::format_descriptor<int32_t>::format()) {
        dtype = c10::ScalarType::Int;
    } else if (buf.format == py::format_descriptor<int64_t>::format()) {
        dtype = c10::ScalarType::Long;
    } else if (buf.format == py::format_descriptor<int8_t>::format()) {
        dtype = c10::ScalarType::Char;  // Char = int8_t
    } else if (buf.format == py::format_descriptor<uint8_t>::format()) {
        dtype = c10::ScalarType::Byte;
    } else if (buf.format == py::format_descriptor<bool>::format() || buf.format == "?") {
        dtype = c10::ScalarType::Bool;
    } else {
        throw std::runtime_error("Unsupported numpy dtype: " + buf.format);
    }

    // Get shape
    std::vector<int64_t> shape(buf.ndim);
    for (size_t i = 0; i < buf.ndim; ++i) {
        shape[i] = buf.shape[i];
    }

    // Create tensor and copy data
    at::Tensor tensor = torch::empty(shape, at::TensorOptions().dtype(dtype).requires_grad(requires_grad));

    // Copy data
    size_t nbytes = buf.size * buf.itemsize;
    std::memcpy(tensor.data_ptr(), buf.ptr, nbytes);

    return tensor;
}

// ============================================================================
// Helper: Convert Tensor to numpy array
// ============================================================================

py::array tensor_to_numpy(const at::Tensor& tensor) {
    // Ensure contiguous and on CPU
    at::Tensor t = tensor.contiguous();
    if (t.device().type() != c10::DeviceType::CPU) {
        throw std::runtime_error("Tensor must be on CPU for numpy conversion");
    }

    // Get shape and strides
    std::vector<py::ssize_t> shape(static_cast<size_t>(t.dim()));
    std::vector<py::ssize_t> strides(static_cast<size_t>(t.dim()));
    for (int64_t i = 0; i < t.dim(); ++i) {
        shape[static_cast<size_t>(i)] = static_cast<py::ssize_t>(t.size(i));
        strides[static_cast<size_t>(i)] = static_cast<py::ssize_t>(t.stride(i) * static_cast<int64_t>(t.itemsize()));
    }

    // Create numpy array based on dtype
    switch (t.dtype()) {
        case c10::ScalarType::Float:
            return py::array(py::dtype::of<float>(), shape, strides,
                static_cast<const float*>(t.data_ptr()));
        case c10::ScalarType::Double:
            return py::array(py::dtype::of<double>(), shape, strides,
                static_cast<const double*>(t.data_ptr()));
        case c10::ScalarType::Int:
            return py::array(py::dtype::of<int32_t>(), shape, strides,
                static_cast<const int32_t*>(t.data_ptr()));
        case c10::ScalarType::Long:
            return py::array(py::dtype::of<int64_t>(), shape, strides,
                static_cast<const int64_t*>(t.data_ptr()));
        case c10::ScalarType::Char:  // int8_t
            return py::array(py::dtype::of<int8_t>(), shape, strides,
                static_cast<const int8_t*>(t.data_ptr()));
        case c10::ScalarType::Byte:
            return py::array(py::dtype::of<uint8_t>(), shape, strides,
                static_cast<const uint8_t*>(t.data_ptr()));
        case c10::ScalarType::Bool:
            return py::array(py::dtype::of<bool>(), shape, strides,
                static_cast<const bool*>(t.data_ptr()));
        default:
            throw std::runtime_error("Unsupported dtype for numpy conversion");
    }
}

// ============================================================================
// Tensor Bindings
// ============================================================================

void init_tensor_bindings(py::module& m) {

    // TensorOptions
    py::class_<at::TensorOptions>(m, "TensorOptions")
        .def(py::init<>())
        .def("dtype", [](at::TensorOptions& self, c10::ScalarType dtype) {
            return self.dtype(dtype);
        })
        .def("device", [](at::TensorOptions& self, const c10::Device& device) {
            return self.device(device);
        })
        .def("requires_grad", [](at::TensorOptions& self, bool requires_grad) {
            return self.requires_grad(requires_grad);
        });

    // Tensor class
    py::class_<at::Tensor>(m, "Tensor")
        // Constructors
        .def(py::init<>())

        // From numpy
        .def(py::init([](py::array arr, bool requires_grad) {
            return numpy_to_tensor(arr, requires_grad);
        }), py::arg("data"), py::arg("requires_grad") = false)

        // Properties
        .def_property_readonly("shape", [](const at::Tensor& t) {
            return std::vector<int64_t>(t.sizes().begin(), t.sizes().end());
        })
        .def("size", [](const at::Tensor& t, py::object dim) -> py::object {
            if (dim.is_none()) {
                return py::cast(std::vector<int64_t>(t.sizes().begin(), t.sizes().end()));
            }
            return py::cast(t.size(dim.cast<int64_t>()));
        }, py::arg("dim") = py::none())
        .def_property_readonly("dtype", &at::Tensor::dtype)
        .def_property_readonly("device", &at::Tensor::device)
        .def_property_readonly("ndim", &at::Tensor::dim)
        .def_property_readonly("dim", &at::Tensor::dim)
        .def_property_readonly("numel", &at::Tensor::numel)
        .def_property_readonly("is_contiguous", [](const at::Tensor& t) { return t.is_contiguous(); })
        .def_property_readonly("requires_grad", [](const at::Tensor& t) { return t.requires_grad(); })
        .def_property_readonly("grad", [](const at::Tensor& t) { return t.grad(); })
        .def_property_readonly("is_leaf", [](const at::Tensor& t) { return t.is_leaf(); })

        // Methods
        .def("numpy", &tensor_to_numpy)
        .def("tolist", [](const at::Tensor& t) {
            return tensor_to_numpy(t.contiguous());
        })
        .def("item", [](const at::Tensor& t) {
            if (t.numel() != 1) {
                throw std::runtime_error("item() only works for single-element tensors");
            }
            return t.item<double>();
        })
        .def("clone", &at::Tensor::clone)
        .def("detach", &at::Tensor::detach)
        .def("contiguous", [](const at::Tensor& t) { return t.contiguous(); })
        .def("requires_grad_", [](at::Tensor& t, bool requires_grad) {
            return t.set_requires_grad(requires_grad);
        }, py::arg("requires_grad") = true)
        .def("zero_", &at::Tensor::zero_)
        .def("fill_", &at::Tensor::fill_, py::arg("value"))
        .def("data_ptr", [](const at::Tensor& t) { return (uintptr_t)t.data_ptr(); })

        // Shape operations
        .def("view", [](const at::Tensor& t, std::vector<int64_t> shape) {
            return t.view(shape);
        })
        .def("reshape", [](const at::Tensor& t, std::vector<int64_t> shape) {
            return t.reshape(shape);
        })
        .def("flatten", [](const at::Tensor& t, int64_t start_dim, int64_t end_dim) {
            return t.flatten(start_dim, end_dim);
        }, py::arg("start_dim") = 0, py::arg("end_dim") = -1)
        .def("squeeze", [](const at::Tensor& t, py::object dim) {
            if (dim.is_none()) {
                return t.squeeze();
            }
            return t.squeeze(dim.cast<int64_t>());
        }, py::arg("dim") = py::none())
        .def("unsqueeze", &at::Tensor::unsqueeze)
        .def("transpose", &at::Tensor::transpose)
        .def("permute", [](const at::Tensor& t, std::vector<int64_t> dims) {
            return t.permute(dims);
        })
        .def("t", &at::Tensor::t)
        .def("expand", [](const at::Tensor& t, std::vector<int64_t> shape) {
            return t.expand(shape);
        })
        .def("repeat", [](const at::Tensor& t, std::vector<int64_t> repeats) {
            return t.repeat(repeats);
        })

        // Math operations
        .def("neg", &at::Tensor::neg)
        .def("abs", &at::Tensor::abs)
        .def("sqrt", &at::Tensor::sqrt)
        .def("rsqrt", &at::Tensor::rsqrt)
        .def("exp", &at::Tensor::exp)
        .def("log", &at::Tensor::log)
        .def("sin", &at::Tensor::sin)
        .def("cos", &at::Tensor::cos)
        .def("tanh", &at::Tensor::tanh)
        .def("sigmoid", &at::Tensor::sigmoid)
        .def("relu", &at::Tensor::relu)
        .def("pow", [](const at::Tensor& t, const at::Tensor& exp) {
            return t.pow(exp);
        })
        .def("pow", [](const at::Tensor& t, double exp) {
            return t.pow(at::Scalar(exp));
        })

        // In-place operations
        .def("neg_", &at::Tensor::neg_)
        .def("abs_", &at::Tensor::abs_)
        .def("sqrt_", &at::Tensor::sqrt_)
        .def("exp_", &at::Tensor::exp_)
        .def("log_", &at::Tensor::log_)
        .def("add_", [](at::Tensor& t, const at::Tensor& other) {
            return t.add_(other);
        })
        .def("sub_", [](at::Tensor& t, const at::Tensor& other) {
            return t.sub_(other);
        })
        .def("mul_", [](at::Tensor& t, const at::Tensor& other) {
            return t.mul_(other);
        })
        .def("div_", [](at::Tensor& t, const at::Tensor& other) {
            return t.div_(other);
        })

        // Reductions
        .def("sum", [](const at::Tensor& t, py::object dim, bool keepdim) {
            if (dim.is_none()) {
                return t.sum();
            }
            return t.sum(dim.cast<int64_t>(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("mean", [](const at::Tensor& t, py::object dim, bool keepdim) {
            if (dim.is_none()) {
                return t.mean();
            }
            return t.mean(dim.cast<int64_t>(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("max", [](const at::Tensor& t, py::object dim, bool keepdim) -> py::object {
            if (dim.is_none()) {
                return py::cast(t.max());
            }
            auto result = t.max(dim.cast<int64_t>(), keepdim);
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("min", [](const at::Tensor& t, py::object dim, bool keepdim) -> py::object {
            if (dim.is_none()) {
                return py::cast(t.min());
            }
            auto result = t.min(dim.cast<int64_t>(), keepdim);
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("argmax", [](const at::Tensor& t, py::object dim, bool keepdim) {
            if (dim.is_none()) {
                return t.argmax();
            }
            return t.argmax(dim.cast<int64_t>(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("argmin", [](const at::Tensor& t, py::object dim, bool keepdim) {
            if (dim.is_none()) {
                return t.argmin();
            }
            return t.argmin(dim.cast<int64_t>(), keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("var", [](const at::Tensor& t, bool unbiased) {
            return t.var(unbiased);
        }, py::arg("unbiased") = true)
        .def("std", [](const at::Tensor& t, bool unbiased) {
            return t.std(unbiased);
        }, py::arg("unbiased") = true)

        // Linear algebra
        .def("mm", &at::Tensor::mm)
        .def("mv", &at::Tensor::mv)
        .def("bmm", &at::Tensor::bmm)
        .def("matmul", &at::Tensor::matmul)
        .def("dot", &at::Tensor::dot)

        // Autograd
        .def("backward", [](at::Tensor& t, py::object gradient,
                            bool retain_graph, bool create_graph) {
            if (gradient.is_none()) {
                torch::autograd::tensor_backward(t, at::Tensor(), retain_graph, create_graph);
            } else {
                torch::autograd::tensor_backward(t, gradient.cast<at::Tensor>(), retain_graph, create_graph);
            }
        }, py::arg("gradient") = py::none(),
           py::arg("retain_graph") = false, py::arg("create_graph") = false)

        // Copy
        .def("copy_", [](at::Tensor& t, const at::Tensor& src) {
            t.copy_(src);
            return t;
        }, py::arg("src"))

        // Type conversion
        .def("to", [](const at::Tensor& t, const std::string& device_or_dtype) {
            // Try as device first
            try {
                c10::Device device(device_or_dtype);
                return t.to(device);
            } catch (...) {}
            throw std::runtime_error("Invalid device or dtype string: " + device_or_dtype);
        }, py::arg("device"))
        .def("to", [](const at::Tensor& t, c10::ScalarType dtype) {
            return t.to(dtype);
        }, py::arg("dtype"))
        .def("to", [](const at::Tensor& t, const c10::Device& device) {
            return t.to(device);
        }, py::arg("device"))
        .def("float", [](const at::Tensor& t) {
            return t.to(c10::ScalarType::Float);
        })
        .def("double", [](const at::Tensor& t) {
            return t.to(c10::ScalarType::Double);
        })
        .def("half", [](const at::Tensor& t) {
            return t.to(c10::ScalarType::Half);
        })
        .def("int", [](const at::Tensor& t) {
            return t.to(c10::ScalarType::Int);
        })
        .def("long", [](const at::Tensor& t) {
            return t.to(c10::ScalarType::Long);
        })
        .def("bool", [](const at::Tensor& t) {
            return t.to(c10::ScalarType::Bool);
        })
        .def("cuda", [](const at::Tensor& t, int device_index) {
            return t.to(c10::Device(c10::DeviceType::CUDA, device_index));
        }, py::arg("device") = 0)
        .def("cpu", [](const at::Tensor& t) {
            return t.to(c10::Device(c10::DeviceType::CPU));
        })

        // Operators
        .def(-py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self + float())
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self < py::self)
        .def(py::self <= py::self)
        .def(py::self > py::self)
        .def(py::self >= py::self)

        // In-place operators (+=, -=, *=, /=)
        .def("__iadd__", [](at::Tensor& t, const at::Tensor& other) -> at::Tensor& {
            t.add_(other);
            return t;
        }, py::return_value_policy::reference)
        .def("__iadd__", [](at::Tensor& t, double val) -> at::Tensor& {
            t.add_(at::Scalar(val));
            return t;
        }, py::return_value_policy::reference)
        .def("__isub__", [](at::Tensor& t, const at::Tensor& other) -> at::Tensor& {
            t.sub_(other);
            return t;
        }, py::return_value_policy::reference)
        .def("__isub__", [](at::Tensor& t, double val) -> at::Tensor& {
            t.sub_(at::Scalar(val));
            return t;
        }, py::return_value_policy::reference)
        .def("__imul__", [](at::Tensor& t, const at::Tensor& other) -> at::Tensor& {
            t.mul_(other);
            return t;
        }, py::return_value_policy::reference)
        .def("__imul__", [](at::Tensor& t, double val) -> at::Tensor& {
            t.mul_(at::Scalar(val));
            return t;
        }, py::return_value_policy::reference)
        .def("__itruediv__", [](at::Tensor& t, const at::Tensor& other) -> at::Tensor& {
            t.div_(other);
            return t;
        }, py::return_value_policy::reference)
        .def("__itruediv__", [](at::Tensor& t, double val) -> at::Tensor& {
            t.div_(at::Scalar(val));
            return t;
        }, py::return_value_policy::reference)

        // Indexing
        .def("__getitem__", [](const at::Tensor& t, int64_t idx) {
            if (idx < 0) idx += t.size(0);
            return t.select(0, idx);
        })
        .def("__getitem__", [](const at::Tensor& t, py::slice slice) {
            size_t start, stop, step, length;
            if (!slice.compute(t.size(0), &start, &stop, &step, &length)) {
                throw py::error_already_set();
            }
            return t.slice(0, start, stop, step);
        })
        .def("__setitem__", [](at::Tensor& t, int64_t idx, const at::Tensor& value) {
            if (idx < 0) idx += t.size(0);
            at::Tensor slice = t.select(0, idx);
            slice.copy_(value);
        })
        .def("__setitem__", [](at::Tensor& t, int64_t idx, double value) {
            if (idx < 0) idx += t.size(0);
            at::Tensor slice = t.select(0, idx);
            slice.fill_(value);
        })
        .def("__len__", [](const at::Tensor& t) {
            return t.size(0);
        })

        // cumsum
        .def("cumsum", [](const at::Tensor& t, int64_t dim) {
            return at::native::cumsum(t, dim);
        }, py::arg("dim"))

        // topk
        .def("topk", [](const at::Tensor& t, int64_t k, int64_t dim, bool largest, bool sorted) {
            auto result = at::native::topk(t, k, dim, largest, sorted);
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        }, py::arg("k"), py::arg("dim") = -1, py::arg("largest") = true, py::arg("sorted") = true)

        // sort
        .def("sort", [](const at::Tensor& t, int64_t dim, bool descending) {
            auto result = at::native::sort(t, dim, descending);
            return py::make_tuple(std::get<0>(result), std::get<1>(result));
        }, py::arg("dim") = -1, py::arg("descending") = false)

        // norm
        .def("norm", [](const at::Tensor& t, py::object p_obj, py::object dim_obj, bool keepdim) -> at::Tensor {
            // Default p=2
            if (dim_obj.is_none()) {
                // Global norm
                at::Tensor sq = t * t;
                return sq.sum().sqrt();
            }
            int64_t dim = dim_obj.cast<int64_t>();
            at::Tensor sq = t * t;
            return sq.sum(dim, keepdim).sqrt();
        }, py::arg("p") = py::none(), py::arg("dim") = py::none(), py::arg("keepdim") = false)

        // type_as
        .def("type_as", [](const at::Tensor& t, const at::Tensor& other) {
            return t.to(other.dtype());
        })

        // scatter
        .def("scatter", [](const at::Tensor& t, int64_t dim, const at::Tensor& index, const at::Tensor& src) {
            at::Tensor result = t.clone();
            // Simple scatter implementation
            if (dim < 0) dim += t.dim();
            int64_t ndim = t.dim();
            if (ndim == 2) {
                for (int64_t i = 0; i < index.size(0); ++i) {
                    for (int64_t j = 0; j < index.size(1); ++j) {
                        int64_t idx = index.data_ptr<int64_t>()[i * index.size(1) + j];
                        float val = src.data_ptr<float>()[i * src.size(1) + j];
                        if (dim == 0) {
                            result.mutable_data_ptr<float>()[idx * result.size(1) + j] = val;
                        } else {
                            result.mutable_data_ptr<float>()[i * result.size(1) + idx] = val;
                        }
                    }
                }
            }
            return result;
        }, py::arg("dim"), py::arg("index"), py::arg("src"))

        // clamp_ (in-place)
        .def("clamp_", [](at::Tensor& t, py::object min_val, py::object max_val) -> at::Tensor& {
            float* d = t.mutable_data_ptr<float>();
            bool has_min = !min_val.is_none();
            bool has_max = !max_val.is_none();
            float mn = has_min ? (float)min_val.cast<double>() : 0.0f;
            float mx = has_max ? (float)max_val.cast<double>() : 0.0f;
            for (int64_t i = 0; i < t.numel(); ++i) {
                if (has_min && d[i] < mn) d[i] = mn;
                if (has_max && d[i] > mx) d[i] = mx;
            }
            return t;
        }, py::arg("min") = py::none(), py::arg("max") = py::none(), py::return_value_policy::reference)

        // clamp (out-of-place) as tensor method
        .def("clamp", [](const at::Tensor& t, py::object min_val, py::object max_val) {
            at::Tensor result = t.clone();
            float* d = result.mutable_data_ptr<float>();
            bool has_min = !min_val.is_none();
            bool has_max = !max_val.is_none();
            float mn = has_min ? (float)min_val.cast<double>() : 0.0f;
            float mx = has_max ? (float)max_val.cast<double>() : 0.0f;
            for (int64_t i = 0; i < result.numel(); ++i) {
                if (has_min && d[i] < mn) d[i] = mn;
                if (has_max && d[i] > mx) d[i] = mx;
            }
            return result;
        }, py::arg("min") = py::none(), py::arg("max") = py::none())

        // String representation
        .def("__repr__", [](const at::Tensor& t) {
            std::ostringstream oss;
            oss << "tensor(";
            // For small tensors (numel <= 20), print actual data
            if (t.numel() <= 20 && t.numel() > 0 && t.dtype() == c10::ScalarType::Float) {
                at::Tensor tc = t.contiguous();
                const float* data = tc.data_ptr<float>();
                if (t.dim() == 0) {
                    oss << data[0];
                } else if (t.dim() == 1) {
                    oss << "[";
                    for (int64_t i = 0; i < t.numel(); ++i) {
                        if (i > 0) oss << ", ";
                        oss << data[i];
                    }
                    oss << "]";
                } else {
                    // Multi-dim: show shape
                    oss << "shape=[";
                    auto sizes = t.sizes();
                    for (size_t i = 0; i < sizes.size(); ++i) {
                        if (i > 0) oss << ", ";
                        oss << sizes[i];
                    }
                    oss << "]";
                }
            } else {
                oss << "shape=[";
                auto sizes = t.sizes();
                for (size_t i = 0; i < sizes.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << sizes[i];
                }
                oss << "]";
            }
            oss << ", dtype=" << c10::toString(t.dtype());
            if (t.device().type() != c10::DeviceType::CPU) {
                oss << ", device=" << t.device().str();
            }
            if (t.requires_grad()) {
                oss << ", requires_grad=True";
            }
            oss << ")";
            return oss.str();
        })
        .def("__str__", [](const at::Tensor& t) {
            std::ostringstream oss;
            oss << t;
            return oss.str();
        });

    // ============================================================================
    // Factory Functions
    // ============================================================================

    m.def("tensor", [](py::array arr, py::object dtype, py::object device, bool requires_grad) {
        at::Tensor t = numpy_to_tensor(arr, requires_grad);
        if (!dtype.is_none()) {
            t = t.to(dtype.cast<c10::ScalarType>());
        }
        if (!device.is_none()) {
            t = t.to(device.cast<c10::Device>());
        }
        return t;
    }, py::arg("data"), py::arg("dtype") = py::none(),
       py::arg("device") = py::none(), py::arg("requires_grad") = false);

    m.def("empty", [](std::vector<int64_t> size, py::object dtype, py::object device, bool requires_grad) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::empty(size, opts);
    }, py::arg("size"), py::arg("dtype") = py::none(),
       py::arg("device") = py::none(), py::arg("requires_grad") = false);

    m.def("zeros", [](std::vector<int64_t> size, py::object dtype, py::object device, bool requires_grad) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::zeros(size, opts);
    }, py::arg("size"), py::arg("dtype") = py::none(),
       py::arg("device") = py::none(), py::arg("requires_grad") = false);

    m.def("ones", [](std::vector<int64_t> size, py::object dtype, py::object device, bool requires_grad) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::ones(size, opts);
    }, py::arg("size"), py::arg("dtype") = py::none(),
       py::arg("device") = py::none(), py::arg("requires_grad") = false);

    m.def("full", [](std::vector<int64_t> size, double fill_value, py::object dtype, py::object device, bool requires_grad) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::full(size, fill_value, opts);
    }, py::arg("size"), py::arg("fill_value"), py::arg("dtype") = py::none(),
       py::arg("device") = py::none(), py::arg("requires_grad") = false);

    m.def("rand", [](std::vector<int64_t> size, py::object dtype, py::object device, bool requires_grad) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::rand(size, opts);
    }, py::arg("size"), py::arg("dtype") = py::none(),
       py::arg("device") = py::none(), py::arg("requires_grad") = false);

    m.def("randn", [](std::vector<int64_t> size, py::object dtype, py::object device, bool requires_grad) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::randn(size, opts);
    }, py::arg("size"), py::arg("dtype") = py::none(),
       py::arg("device") = py::none(), py::arg("requires_grad") = false);

    m.def("randint", [](int64_t low, int64_t high, std::vector<int64_t> size, py::object dtype, py::object device) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        else opts = opts.dtype(c10::ScalarType::Long);
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        return torch::randint(low, high, size, opts);
    }, py::arg("low"), py::arg("high"), py::arg("size"),
       py::arg("dtype") = py::none(), py::arg("device") = py::none());

    m.def("arange", [](double start, double end, double step, py::object dtype, py::object device, bool requires_grad) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::arange(start, end, step, opts);
    }, py::arg("start"), py::arg("end"), py::arg("step") = 1.0,
       py::arg("dtype") = py::none(), py::arg("device") = py::none(),
       py::arg("requires_grad") = false);

    m.def("linspace", [](double start, double end, int64_t steps, py::object dtype, py::object device, bool requires_grad) {
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::linspace(start, end, steps, opts);
    }, py::arg("start"), py::arg("end"), py::arg("steps"),
       py::arg("dtype") = py::none(), py::arg("device") = py::none(),
       py::arg("requires_grad") = false);

    m.def("eye", [](int64_t n, py::object m, py::object dtype, py::object device, bool requires_grad) {
        int64_t cols = m.is_none() ? n : m.cast<int64_t>();
        at::TensorOptions opts;
        if (!dtype.is_none()) opts = opts.dtype(dtype.cast<c10::ScalarType>());
        if (!device.is_none()) opts = opts.device(device.cast<c10::Device>());
        opts = opts.requires_grad(requires_grad);
        return torch::eye(n, cols, opts);
    }, py::arg("n"), py::arg("m") = py::none(), py::arg("dtype") = py::none(),
       py::arg("device") = py::none(), py::arg("requires_grad") = false);

    // ============================================================================
    // Operations on Tensors
    // ============================================================================

    m.def("cat", &torch::cat, py::arg("tensors"), py::arg("dim") = 0);
    m.def("stack", &torch::stack, py::arg("tensors"), py::arg("dim") = 0);
    m.def("split", [](const at::Tensor& t, int64_t split_size, int64_t dim) {
        return t.split(split_size, dim);
    }, py::arg("tensor"), py::arg("split_size"), py::arg("dim") = 0);
    m.def("chunk", [](const at::Tensor& t, int64_t chunks, int64_t dim) {
        return t.chunk(chunks, dim);
    }, py::arg("tensor"), py::arg("chunks"), py::arg("dim") = 0);

    m.def("mm", &torch::mm);
    m.def("bmm", &torch::bmm);
    m.def("matmul", &torch::matmul);
    m.def("dot", [](const at::Tensor& a, const at::Tensor& b) {
        return a.dot(b);
    });

    m.def("sum", [](const at::Tensor& t, py::object dim, bool keepdim) {
        if (dim.is_none()) return t.sum();
        return t.sum(dim.cast<int64_t>(), keepdim);
    }, py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    m.def("mean", [](const at::Tensor& t, py::object dim, bool keepdim) {
        if (dim.is_none()) return t.mean();
        return t.mean(dim.cast<int64_t>(), keepdim);
    }, py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    m.def("max", [](const at::Tensor& t) { return t.max(); });
    m.def("min", [](const at::Tensor& t) { return t.min(); });

    m.def("sqrt", [](const at::Tensor& t) { return t.sqrt(); });
    m.def("exp", [](const at::Tensor& t) { return t.exp(); });
    m.def("log", [](const at::Tensor& t) { return t.log(); });
    m.def("sin", [](const at::Tensor& t) { return t.sin(); });
    m.def("cos", [](const at::Tensor& t) { return t.cos(); });
    m.def("tanh", [](const at::Tensor& t) { return t.tanh(); });
    m.def("sigmoid", [](const at::Tensor& t) { return t.sigmoid(); });
    m.def("relu", [](const at::Tensor& t) { return t.relu(); });
    m.def("softmax", [](const at::Tensor& t, int64_t dim) {
        // CPU softmax implementation
        // softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        int64_t ndim = t.dim();
        if (dim < 0) dim += ndim;

        // For simplicity, use manual computation
        auto max_result = t.max(dim, true);
        at::Tensor max_vals = std::get<0>(max_result);
        at::Tensor shifted = t - max_vals;
        at::Tensor exp_vals = shifted.exp();
        at::Tensor sum_exp = exp_vals.sum(dim, true);
        return exp_vals / sum_exp;
    }, py::arg("input"), py::arg("dim"));

    m.def("clamp", [](const at::Tensor& t, py::object min_val, py::object max_val) {
        // Implement clamp: max(min_v, min(x, max_v))
        at::Tensor result = t.clone();
        bool has_min = !min_val.is_none();
        bool has_max = !max_val.is_none();

        if (has_min && has_max) {
            double min_v = min_val.cast<double>();
            double max_v = max_val.cast<double>();
            // clamp to both bounds
            at::Tensor min_tensor = torch::full(t.sizes(), min_v, at::TensorOptions().dtype(t.dtype()));
            at::Tensor max_tensor = torch::full(t.sizes(), max_v, at::TensorOptions().dtype(t.dtype()));
            result = at::maximum(min_tensor, at::minimum(result, max_tensor));
        } else if (has_min) {
            double min_v = min_val.cast<double>();
            at::Tensor min_tensor = torch::full(t.sizes(), min_v, at::TensorOptions().dtype(t.dtype()));
            result = at::maximum(min_tensor, result);
        } else if (has_max) {
            double max_v = max_val.cast<double>();
            at::Tensor max_tensor = torch::full(t.sizes(), max_v, at::TensorOptions().dtype(t.dtype()));
            result = at::minimum(result, max_tensor);
        }
        return result;
    }, py::arg("input"), py::arg("min") = py::none(), py::arg("max") = py::none());

    m.def("where", [](const at::Tensor& condition, const at::Tensor& x, const at::Tensor& y) {
        return torch::where(condition, x, y);
    });

    // ============================================================================
    // Additional Operations for PIR model support
    // ============================================================================

    // cumsum
    m.def("cumsum", [](const at::Tensor& t, int64_t dim) {
        return at::native::cumsum(t, dim);
    }, py::arg("input"), py::arg("dim"));

    // rsqrt (module-level)
    m.def("rsqrt", [](const at::Tensor& t) { return t.rsqrt(); });

    // norm
    m.def("norm", [](const at::Tensor& t, py::object p_obj, py::object dim_obj, bool keepdim) -> at::Tensor {
        if (dim_obj.is_none()) {
            at::Tensor sq = t * t;
            return sq.sum().sqrt();
        }
        int64_t dim = dim_obj.cast<int64_t>();
        at::Tensor sq = t * t;
        return sq.sum(dim, keepdim).sqrt();
    }, py::arg("input"), py::arg("p") = py::none(), py::arg("dim") = py::none(), py::arg("keepdim") = false);

    // topk
    m.def("topk", [](const at::Tensor& t, int64_t k, int64_t dim, bool largest, bool sorted) {
        auto result = at::native::topk(t, k, dim, largest, sorted);
        return py::make_tuple(std::get<0>(result), std::get<1>(result));
    }, py::arg("input"), py::arg("k"), py::arg("dim") = -1,
       py::arg("largest") = true, py::arg("sorted") = true);

    // sort
    m.def("sort", [](const at::Tensor& t, int64_t dim, bool descending) {
        auto result = at::native::sort(t, dim, descending);
        return py::make_tuple(std::get<0>(result), std::get<1>(result));
    }, py::arg("input"), py::arg("dim") = -1, py::arg("descending") = false);

    // zeros_like
    m.def("zeros_like", [](const at::Tensor& t) {
        return torch::zeros(t.sizes(), at::TensorOptions().dtype(t.dtype()));
    }, py::arg("input"));

    // ones_like
    m.def("ones_like", [](const at::Tensor& t) {
        return torch::ones(t.sizes(), at::TensorOptions().dtype(t.dtype()));
    }, py::arg("input"));

    // from_numpy
    m.def("from_numpy", [](py::array arr) {
        return numpy_to_tensor(arr, false);
    }, py::arg("ndarray"));

    // nan_to_num
    m.def("nan_to_num", [](const at::Tensor& t, double nan_val, double posinf_val, double neginf_val) {
        at::Tensor result = t.clone();
        float* d = result.mutable_data_ptr<float>();
        float nan_f = (float)nan_val;
        float posinf_f = (float)posinf_val;
        float neginf_f = (float)neginf_val;
        for (int64_t i = 0; i < result.numel(); ++i) {
            if (std::isnan(d[i])) d[i] = nan_f;
            else if (std::isinf(d[i]) && d[i] > 0) d[i] = posinf_f;
            else if (std::isinf(d[i]) && d[i] < 0) d[i] = neginf_f;
        }
        return result;
    }, py::arg("input"), py::arg("nan") = 0.0, py::arg("posinf") = 1e4, py::arg("neginf") = -1e4);

    // isinf
    m.def("isinf", [](const at::Tensor& t) {
        at::Tensor result = torch::zeros(t.sizes(), at::TensorOptions().dtype(c10::ScalarType::Bool));
        const float* src = t.data_ptr<float>();
        bool* dst = result.mutable_data_ptr<bool>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            dst[i] = std::isinf(src[i]);
        }
        return result;
    }, py::arg("input"));

    // isnan
    m.def("isnan", [](const at::Tensor& t) {
        at::Tensor result = torch::zeros(t.sizes(), at::TensorOptions().dtype(c10::ScalarType::Bool));
        const float* src = t.data_ptr<float>();
        bool* dst = result.mutable_data_ptr<bool>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            dst[i] = std::isnan(src[i]);
        }
        return result;
    }, py::arg("input"));

    // isfinite
    m.def("isfinite", [](const at::Tensor& t) {
        at::Tensor result = torch::zeros(t.sizes(), at::TensorOptions().dtype(c10::ScalarType::Bool));
        const float* src = t.data_ptr<float>();
        bool* dst = result.mutable_data_ptr<bool>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            dst[i] = std::isfinite(src[i]);
        }
        return result;
    }, py::arg("input"));

    // multinomial
    m.def("multinomial", [](const at::Tensor& t, int64_t num_samples, bool replacement) {
        // Simple multinomial sampling from probability distribution
        // t is [batch, num_classes] or [num_classes]
        bool batched = t.dim() == 2;
        int64_t batch_size = batched ? t.size(0) : 1;
        int64_t num_classes = batched ? t.size(1) : t.size(0);

        std::vector<int64_t> out_shape;
        if (batched) {
            out_shape = {batch_size, num_samples};
        } else {
            out_shape = {num_samples};
        }
        at::Tensor result = torch::zeros(out_shape, at::TensorOptions().dtype(c10::ScalarType::Long));
        int64_t* out_data = result.mutable_data_ptr<int64_t>();

        at::Tensor tc = t.contiguous();
        const float* probs = tc.data_ptr<float>();

        for (int64_t b = 0; b < batch_size; ++b) {
            const float* row = probs + b * num_classes;
            for (int64_t s = 0; s < num_samples; ++s) {
                double r = (double)::rand() / RAND_MAX;
                double cumulative = 0.0;
                int64_t chosen = num_classes - 1;
                for (int64_t c = 0; c < num_classes; ++c) {
                    cumulative += (double)row[c];
                    if (r < cumulative) {
                        chosen = c;
                        break;
                    }
                }
                out_data[b * num_samples + s] = chosen;
            }
        }
        return result;
    }, py::arg("input"), py::arg("num_samples"), py::arg("replacement") = false);

    // einsum (basic: supports "i,j->ij" outer product and common patterns)
    m.def("einsum", [](const std::string& equation, py::args tensors) {
        // Parse simple patterns used in PIR model
        if (tensors.size() == 2) {
            at::Tensor a = tensors[0].cast<at::Tensor>();
            at::Tensor b = tensors[1].cast<at::Tensor>();

            if (equation == "i,j->ij") {
                // Outer product
                return a.unsqueeze(1).mm(b.unsqueeze(0));
            }
            // For other patterns, delegate to native einsum
            return at::native::einsum(equation, {a, b});
        } else if (tensors.size() == 1) {
            return tensors[0].cast<at::Tensor>();
        }
        throw std::runtime_error("einsum: unsupported number of tensors: " + std::to_string(tensors.size()));
    }, py::arg("equation"));

    // compile (no-op: just returns the model unchanged)
    m.def("compile", [](py::object model, py::kwargs kwargs) {
        return model;
    }, py::arg("model"));

    // Note: no_grad context manager is defined in autograd_bindings.cpp
}
