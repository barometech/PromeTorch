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
#include "torch/serialization.h"
#include "torch/data/data.h"

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
// Serialization Bindings
// ============================================================================

void init_serialization_bindings(py::module& m) {
    m.def("save", [](const at::Tensor& tensor, const std::string& path) {
        torch::save(tensor, path);
    }, py::arg("tensor"), py::arg("path"),
    "Save a tensor to a file in PTOR binary format");

    m.def("load", [](const std::string& path) {
        return torch::load(path);
    }, py::arg("path"),
    "Load a tensor from a PTOR binary file");

    m.def("save_state_dict", [](const std::unordered_map<std::string, at::Tensor>& state_dict,
                                 const std::string& path) {
        torch::save_state_dict(state_dict, path);
    }, py::arg("state_dict"), py::arg("path"),
    "Save a state dict (name->tensor map) to a file");

    m.def("load_state_dict", [](const std::string& path) {
        return torch::load_state_dict(path);
    }, py::arg("path"),
    "Load a state dict from a file");
}

// ============================================================================
// Data Loading Bindings
// ============================================================================

void init_data_bindings(py::module& m) {
    using TDS = torch::data::TensorDataset;
    using Batch = torch::data::Batch<at::Tensor, at::Tensor>;
    using DL = torch::data::DataLoader<TDS>;

    // Batch type
    py::class_<Batch>(m, "Batch")
        .def_readonly("data", &Batch::data)
        .def_readonly("target", &Batch::target)
        .def_readonly("size", &Batch::size);

    // TensorDataset
    py::class_<TDS, std::shared_ptr<TDS>>(m, "TensorDataset")
        .def(py::init<at::Tensor, at::Tensor>(), py::arg("data"), py::arg("targets"))
        .def(py::init<at::Tensor>(), py::arg("data"))
        .def("__len__", &TDS::size)
        .def("__getitem__", [](TDS& self, size_t index) {
            auto ex = self.get(index);
            return py::make_tuple(ex.data, ex.target);
        });

    // DataLoaderOptions
    py::class_<torch::data::DataLoaderOptions>(m, "DataLoaderOptions")
        .def(py::init<>())
        .def_readwrite("batch_size", &torch::data::DataLoaderOptions::batch_size)
        .def_readwrite("shuffle", &torch::data::DataLoaderOptions::shuffle)
        .def_readwrite("drop_last", &torch::data::DataLoaderOptions::drop_last)
        .def_readwrite("num_workers", &torch::data::DataLoaderOptions::num_workers);

    // DataLoader for TensorDataset
    py::class_<DL>(m, "DataLoader")
        .def(py::init([](std::shared_ptr<TDS> dataset, size_t batch_size,
                         bool shuffle, bool drop_last) {
            torch::data::DataLoaderOptions opts;
            opts.batch_size = batch_size;
            opts.shuffle = shuffle;
            opts.drop_last = drop_last;
            return std::make_unique<DL>(*dataset, opts);
        }), py::arg("dataset"), py::arg("batch_size") = 1,
            py::arg("shuffle") = false, py::arg("drop_last") = false)
        .def("__len__", &DL::size)
        .def("__iter__", [](DL& self) {
            // Collect all batches into a list for Python iteration
            // This is simpler than exposing C++ iterators to Python
            py::list batches;
            for (auto it = self.begin(); it != self.end(); ++it) {
                auto& batch = *it;
                batches.append(py::make_tuple(batch.data, batch.target));
            }
            return batches.attr("__iter__")();
        });
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

    // Serialization
    init_serialization_bindings(m);

    // Create submodules
    py::module nn = m.def_submodule("nn", "Neural Network modules");
    init_nn_bindings(nn);

    py::module optim = m.def_submodule("optim", "Optimizers");
    init_optim_bindings(optim);

    // Data loading submodule
    py::module data = m.def_submodule("data", "Data loading utilities");
    init_data_bindings(data);

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
    m.attr("__version__") = "0.2.0";
}
