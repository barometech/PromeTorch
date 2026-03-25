// ============================================================================
// PromeTorch Python Bindings - Autograd
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"
#include "torch/csrc/autograd/grad_mode.h"

namespace py = pybind11;

// ============================================================================
// Python-friendly wrappers around real C++ GradMode guards
// ============================================================================
// We need thin wrappers because the C++ guards are non-copyable RAII objects.
// These wrappers delegate to torch::autograd::GradMode directly.
//
// BUG-C9 FIX: Added `restored_` flag to prevent double-restore.
// Without it, __exit__() restores prev_enabled_, then the destructor
// restores it AGAIN — which can flip grad mode to the wrong state if
// another guard was entered/exited between __exit__ and destruction.

class PyNoGradGuard {
public:
    PyNoGradGuard()
        : prev_enabled_(torch::autograd::GradMode::is_enabled())
        , restored_(false) {
        torch::autograd::GradMode::set_enabled(false);
    }

    ~PyNoGradGuard() {
        if (!restored_) {
            torch::autograd::GradMode::set_enabled(prev_enabled_);
        }
    }

    // __enter__ must return self for Python context manager protocol
    PyNoGradGuard& enter() { return *this; }

    void exit(py::object, py::object, py::object) {
        if (!restored_) {
            torch::autograd::GradMode::set_enabled(prev_enabled_);
            restored_ = true;
        }
    }

private:
    bool prev_enabled_;
    bool restored_;
};

class PyEnableGradGuard {
public:
    PyEnableGradGuard()
        : prev_enabled_(torch::autograd::GradMode::is_enabled())
        , restored_(false) {
        torch::autograd::GradMode::set_enabled(true);
    }

    ~PyEnableGradGuard() {
        if (!restored_) {
            torch::autograd::GradMode::set_enabled(prev_enabled_);
        }
    }

    // __enter__ must return self for Python context manager protocol
    PyEnableGradGuard& enter() { return *this; }

    void exit(py::object, py::object, py::object) {
        if (!restored_) {
            torch::autograd::GradMode::set_enabled(prev_enabled_);
            restored_ = true;
        }
    }

private:
    bool prev_enabled_;
    bool restored_;
};

// ============================================================================
// Autograd Bindings
// ============================================================================

void init_autograd_bindings(py::module& m) {

    // GradMode - delegates to the REAL C++ torch::autograd::GradMode
    py::class_<torch::autograd::GradMode>(m, "GradMode")
        .def_static("is_enabled", &torch::autograd::GradMode::is_enabled)
        .def_static("set_enabled", &torch::autograd::GradMode::set_enabled);

    // no_grad context manager
    py::class_<PyNoGradGuard>(m, "no_grad")
        .def(py::init<>())
        .def("__enter__", &PyNoGradGuard::enter, py::return_value_policy::reference)
        .def("__exit__", &PyNoGradGuard::exit);

    // enable_grad context manager
    py::class_<PyEnableGradGuard>(m, "enable_grad")
        .def(py::init<>())
        .def("__enter__", &PyEnableGradGuard::enter, py::return_value_policy::reference)
        .def("__exit__", &PyEnableGradGuard::exit);

    // is_grad_enabled - delegates to real GradMode
    m.def("is_grad_enabled", &torch::autograd::GradMode::is_enabled);
    m.def("set_grad_enabled", &torch::autograd::GradMode::set_enabled);

    // backward function - uses tensor_backward from autograd.h
    m.def("backward", [](const at::Tensor& tensor, py::object grad_tensor,
                         bool retain_graph, bool create_graph) {
        if (grad_tensor.is_none()) {
            torch::autograd::tensor_backward(tensor, at::Tensor(), retain_graph, create_graph);
        } else {
            torch::autograd::tensor_backward(tensor, grad_tensor.cast<at::Tensor>(), retain_graph, create_graph);
        }
    }, py::arg("tensor"), py::arg("grad_tensor") = py::none(),
       py::arg("retain_graph") = false, py::arg("create_graph") = false);

    // grad function (compute gradients without accumulating)
    m.def("grad", [](const at::Tensor& output, const std::vector<at::Tensor>& inputs,
                     py::object grad_outputs, bool retain_graph, bool create_graph) {
        std::vector<at::Tensor> grad_out;
        if (!grad_outputs.is_none()) {
            if (py::isinstance<at::Tensor>(grad_outputs)) {
                grad_out.push_back(grad_outputs.cast<at::Tensor>());
            } else {
                grad_out = grad_outputs.cast<std::vector<at::Tensor>>();
            }
        }
        return torch::autograd::grad({output}, inputs, grad_out, retain_graph, create_graph);
    }, py::arg("outputs"), py::arg("inputs"), py::arg("grad_outputs") = py::none(),
       py::arg("retain_graph") = false, py::arg("create_graph") = false);
}
