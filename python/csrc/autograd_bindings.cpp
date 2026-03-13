// ============================================================================
// PromeTorch Python Bindings - Autograd
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"

namespace py = pybind11;

// ============================================================================
// Simple GradMode implementation (global state)
// ============================================================================

namespace {
    thread_local bool grad_enabled_ = true;
}

class GradMode {
public:
    static bool is_enabled() { return grad_enabled_; }
    static void set_enabled(bool enabled) { grad_enabled_ = enabled; }
};

// ============================================================================
// NoGrad Context Manager
// ============================================================================

class NoGradGuard {
public:
    NoGradGuard() : prev_enabled_(GradMode::is_enabled()) {
        GradMode::set_enabled(false);
    }

    ~NoGradGuard() {
        GradMode::set_enabled(prev_enabled_);
    }

    void enter() {}
    void exit(py::object, py::object, py::object) {
        GradMode::set_enabled(prev_enabled_);
    }

private:
    bool prev_enabled_;
};

// ============================================================================
// EnableGrad Context Manager
// ============================================================================

class EnableGradGuard {
public:
    EnableGradGuard() : prev_enabled_(GradMode::is_enabled()) {
        GradMode::set_enabled(true);
    }

    ~EnableGradGuard() {
        GradMode::set_enabled(prev_enabled_);
    }

    void enter() {}
    void exit(py::object, py::object, py::object) {
        GradMode::set_enabled(prev_enabled_);
    }

private:
    bool prev_enabled_;
};

// ============================================================================
// Autograd Bindings
// ============================================================================

void init_autograd_bindings(py::module& m) {

    // GradMode
    py::class_<GradMode>(m, "GradMode")
        .def_static("is_enabled", &GradMode::is_enabled)
        .def_static("set_enabled", &GradMode::set_enabled);

    // no_grad context manager
    py::class_<NoGradGuard>(m, "no_grad")
        .def(py::init<>())
        .def("__enter__", &NoGradGuard::enter)
        .def("__exit__", &NoGradGuard::exit);

    // enable_grad context manager
    py::class_<EnableGradGuard>(m, "enable_grad")
        .def(py::init<>())
        .def("__enter__", &EnableGradGuard::enter)
        .def("__exit__", &EnableGradGuard::exit);

    // is_grad_enabled
    m.def("is_grad_enabled", &GradMode::is_enabled);
    m.def("set_grad_enabled", &GradMode::set_enabled);

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
