// ============================================================================
// PromeTorch Python Bindings - Main Entry Point
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"

namespace py = pybind11;

// Forward declarations
void init_tensor_bindings(py::module& m);
void init_autograd_bindings(py::module& m);
void init_nn_bindings(py::module& m);
void init_optim_bindings(py::module& m);

// ============================================================================
// Device Bindings
// ============================================================================

void init_device_bindings(py::module& m) {
    py::enum_<c10::DeviceType>(m, "DeviceType")
        .value("CPU", c10::DeviceType::CPU)
        .value("CUDA", c10::DeviceType::CUDA)
        .export_values();

    py::class_<c10::Device>(m, "device")
        .def(py::init<c10::DeviceType, c10::DeviceIndex>(),
             py::arg("type"), py::arg("index") = 0)
        .def(py::init([](const std::string& s) {
            return c10::Device(s);
        }), py::arg("device_string"))
        .def_property_readonly("type", &c10::Device::type)
        .def_property_readonly("index", &c10::Device::index)
        .def("__repr__", [](const c10::Device& d) {
            return d.str();
        })
        .def("__eq__", &c10::Device::operator==)
        .def("__ne__", &c10::Device::operator!=);
}

// ============================================================================
// ScalarType Bindings
// ============================================================================

void init_dtype_bindings(py::module& m) {
    py::enum_<c10::ScalarType>(m, "dtype")
        .value("float16", c10::ScalarType::Half)
        .value("float32", c10::ScalarType::Float)
        .value("float64", c10::ScalarType::Double)
        .value("bfloat16", c10::ScalarType::BFloat16)
        .value("int8", c10::ScalarType::Char)      // Char = int8_t
        .value("int16", c10::ScalarType::Short)    // Short = int16_t
        .value("int32", c10::ScalarType::Int)
        .value("int64", c10::ScalarType::Long)
        .value("uint8", c10::ScalarType::Byte)
        .value("bool", c10::ScalarType::Bool)
        .value("complex64", c10::ScalarType::ComplexFloat)
        .value("complex128", c10::ScalarType::ComplexDouble)
        .export_values();

    // Convenience aliases
    m.attr("half") = c10::ScalarType::Half;
    m.attr("float") = c10::ScalarType::Float;
    m.attr("double") = c10::ScalarType::Double;
    m.attr("bfloat16") = c10::ScalarType::BFloat16;
    m.attr("int8") = c10::ScalarType::Char;
    m.attr("int16") = c10::ScalarType::Short;
    m.attr("int32") = c10::ScalarType::Int;
    m.attr("int64") = c10::ScalarType::Long;
    m.attr("uint8") = c10::ScalarType::Byte;
    m.attr("bool") = c10::ScalarType::Bool;
    m.attr("long") = c10::ScalarType::Long;
}

// ============================================================================
// Main Module
// ============================================================================

PYBIND11_MODULE(_C, m) {
    m.doc() = "PromeTorch: A PyTorch-like Deep Learning Framework";

    // Initialize all bindings
    init_device_bindings(m);
    init_dtype_bindings(m);
    init_tensor_bindings(m);
    init_autograd_bindings(m);

    // Create submodules
    py::module nn = m.def_submodule("nn", "Neural Network modules");
    init_nn_bindings(nn);

    py::module optim = m.def_submodule("optim", "Optimizers");
    init_optim_bindings(optim);

    // CUDA availability
    m.def("cuda_is_available", []() {
#ifdef PT_USE_CUDA
        return true;
#else
        return false;
#endif
    }, "Check if CUDA is available");

    m.def("cuda_device_count", []() {
#ifdef PT_USE_CUDA
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
#else
        return 0;
#endif
    }, "Get number of CUDA devices");

    // Version info
    m.attr("__version__") = "0.1.0";
}
