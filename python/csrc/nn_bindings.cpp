// ============================================================================
// PromeTorch Python Bindings - Neural Network (Simplified)
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <sstream>
#include <cmath>

#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"

namespace py = pybind11;

// ============================================================================
// NN Module Bindings (Simplified)
// ============================================================================

void init_nn_bindings(py::module& m) {

    // Base Module class
    py::class_<torch::nn::Module, std::shared_ptr<torch::nn::Module>>(m, "Module")
        .def("train", [](torch::nn::Module& self, bool mode) {
            self.train(mode);
        }, py::arg("mode") = true)
        .def("eval", &torch::nn::Module::eval)
        .def("zero_grad", &torch::nn::Module::zero_grad)
        .def("parameters", [](torch::nn::Module& self) {
            std::vector<at::Tensor> params;
            for (auto* p : self.parameters()) {  // p is a pointer
                params.push_back(p->data());
            }
            return params;
        })
        .def("named_parameters", [](torch::nn::Module& self) {
            std::vector<std::pair<std::string, at::Tensor>> params;
            for (auto& np : self.named_parameters()) {
                // np.first is string, np.second is Parameter*
                params.push_back(std::make_pair(np.first, np.second->data()));
            }
            return params;
        })
        .def("name", &torch::nn::Module::name)
        .def("is_training", &torch::nn::Module::is_training);

    // Linear layer
    py::class_<torch::nn::Linear, torch::nn::Module, std::shared_ptr<torch::nn::Linear>>(m, "Linear")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &torch::nn::Linear::forward)
        .def("__call__", &torch::nn::Linear::forward)
        .def_property_readonly("weight", [](torch::nn::Linear& self) {
            auto* p = self.get_parameter("weight");
            return p ? p->data() : at::Tensor();
        })
        .def_property_readonly("bias", [](torch::nn::Linear& self) -> py::object {
            auto* p = self.get_parameter("bias");
            if (p && p->data().defined()) {
                return py::cast(p->data());
            }
            return py::none();
        });

    // Activation functions as modules
    py::class_<torch::nn::ReLU, torch::nn::Module, std::shared_ptr<torch::nn::ReLU>>(m, "ReLU")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def("forward", &torch::nn::ReLU::forward)
        .def("__call__", &torch::nn::ReLU::forward);

    py::class_<torch::nn::Sigmoid, torch::nn::Module, std::shared_ptr<torch::nn::Sigmoid>>(m, "Sigmoid")
        .def(py::init<>())
        .def("forward", &torch::nn::Sigmoid::forward)
        .def("__call__", &torch::nn::Sigmoid::forward);

    py::class_<torch::nn::Tanh, torch::nn::Module, std::shared_ptr<torch::nn::Tanh>>(m, "Tanh")
        .def(py::init<>())
        .def("forward", &torch::nn::Tanh::forward)
        .def("__call__", &torch::nn::Tanh::forward);

    py::class_<torch::nn::Softmax, torch::nn::Module, std::shared_ptr<torch::nn::Softmax>>(m, "Softmax")
        .def(py::init<int64_t>(), py::arg("dim") = -1)
        .def("forward", &torch::nn::Softmax::forward)
        .def("__call__", &torch::nn::Softmax::forward);

    py::class_<torch::nn::GELU, torch::nn::Module, std::shared_ptr<torch::nn::GELU>>(m, "GELU")
        .def(py::init<std::string>(), py::arg("approximate") = "none")
        .def("forward", &torch::nn::GELU::forward)
        .def("__call__", &torch::nn::GELU::forward);

    py::class_<torch::nn::SiLU, torch::nn::Module, std::shared_ptr<torch::nn::SiLU>>(m, "SiLU")
        .def(py::init<>())
        .def("forward", &torch::nn::SiLU::forward)
        .def("__call__", &torch::nn::SiLU::forward);

    // Dropout
    py::class_<torch::nn::Dropout, torch::nn::Module, std::shared_ptr<torch::nn::Dropout>>(m, "Dropout")
        .def(py::init<double, bool>(), py::arg("p") = 0.5, py::arg("inplace") = false)
        .def("forward", &torch::nn::Dropout::forward)
        .def("__call__", &torch::nn::Dropout::forward);

    // BatchNorm2d
    py::class_<torch::nn::BatchNorm2d, torch::nn::Module, std::shared_ptr<torch::nn::BatchNorm2d>>(m, "BatchNorm2d")
        .def(py::init<int64_t, double, double, bool, bool>(),
             py::arg("num_features"), py::arg("eps") = 1e-5, py::arg("momentum") = 0.1,
             py::arg("affine") = true, py::arg("track_running_stats") = true)
        .def("forward", &torch::nn::BatchNorm2d::forward)
        .def("__call__", &torch::nn::BatchNorm2d::forward);

    // LayerNorm
    py::class_<torch::nn::LayerNorm, torch::nn::Module, std::shared_ptr<torch::nn::LayerNorm>>(m, "LayerNorm")
        .def(py::init<std::vector<int64_t>, double, bool>(),
             py::arg("normalized_shape"), py::arg("eps") = 1e-5, py::arg("elementwise_affine") = true)
        .def("forward", &torch::nn::LayerNorm::forward)
        .def("__call__", &torch::nn::LayerNorm::forward);

    // Conv2d
    py::class_<torch::nn::Conv2d, torch::nn::Module, std::shared_ptr<torch::nn::Conv2d>>(m, "Conv2d")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>(),
             py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
             py::arg("stride") = 1, py::arg("padding") = 0, py::arg("dilation") = 1,
             py::arg("groups") = 1, py::arg("bias") = true)
        .def("forward", &torch::nn::Conv2d::forward)
        .def("__call__", &torch::nn::Conv2d::forward)
        .def_property_readonly("weight", [](torch::nn::Conv2d& self) {
            auto* p = self.get_parameter("weight");
            return p ? p->data() : at::Tensor();
        });

    // Embedding
    py::class_<torch::nn::Embedding, torch::nn::Module, std::shared_ptr<torch::nn::Embedding>>(m, "Embedding")
        .def(py::init<int64_t, int64_t, int64_t, double, bool>(),
             py::arg("num_embeddings"), py::arg("embedding_dim"),
             py::arg("padding_idx") = -1, py::arg("max_norm") = 0.0, py::arg("sparse") = false)
        .def("forward", &torch::nn::Embedding::forward)
        .def("__call__", &torch::nn::Embedding::forward)
        .def_property_readonly("weight", [](torch::nn::Embedding& self) {
            auto* p = self.get_parameter("weight");
            return p ? p->data() : at::Tensor();
        });

    // MaxPool2d
    py::class_<torch::nn::MaxPool2d, torch::nn::Module, std::shared_ptr<torch::nn::MaxPool2d>>(m, "MaxPool2d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"), py::arg("stride") = 0, py::arg("padding") = 0)
        .def("forward", &torch::nn::MaxPool2d::forward)
        .def("__call__", &torch::nn::MaxPool2d::forward);

    // AvgPool2d
    py::class_<torch::nn::AvgPool2d, torch::nn::Module, std::shared_ptr<torch::nn::AvgPool2d>>(m, "AvgPool2d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"), py::arg("stride") = 0, py::arg("padding") = 0)
        .def("forward", &torch::nn::AvgPool2d::forward)
        .def("__call__", &torch::nn::AvgPool2d::forward);

    // Reduction enum - must be defined before loss classes that use it
    py::enum_<torch::nn::Reduction>(m, "Reduction")
        .value("None_", torch::nn::Reduction::None)
        .value("Mean", torch::nn::Reduction::Mean)
        .value("Sum", torch::nn::Reduction::Sum)
        .export_values();

    // Loss functions - using __call__ with 2 arguments
    py::class_<torch::nn::MSELoss, torch::nn::Module, std::shared_ptr<torch::nn::MSELoss>>(m, "MSELoss")
        .def(py::init<torch::nn::Reduction>(), py::arg("reduction") = torch::nn::Reduction::Mean)
        .def("__call__", [](torch::nn::MSELoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    py::class_<torch::nn::CrossEntropyLoss, torch::nn::Module, std::shared_ptr<torch::nn::CrossEntropyLoss>>(m, "CrossEntropyLoss")
        .def(py::init<torch::nn::Reduction, int64_t, double>(),
             py::arg("reduction") = torch::nn::Reduction::Mean,
             py::arg("ignore_index") = -100, py::arg("label_smoothing") = 0.0)
        .def("__call__", [](torch::nn::CrossEntropyLoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    py::class_<torch::nn::NLLLoss, torch::nn::Module, std::shared_ptr<torch::nn::NLLLoss>>(m, "NLLLoss")
        .def(py::init<torch::nn::Reduction, int64_t>(),
             py::arg("reduction") = torch::nn::Reduction::Mean, py::arg("ignore_index") = -100)
        .def("__call__", [](torch::nn::NLLLoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    py::class_<torch::nn::BCELoss, torch::nn::Module, std::shared_ptr<torch::nn::BCELoss>>(m, "BCELoss")
        .def(py::init<torch::nn::Reduction>(), py::arg("reduction") = torch::nn::Reduction::Mean)
        .def("__call__", [](torch::nn::BCELoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    py::class_<torch::nn::L1Loss, torch::nn::Module, std::shared_ptr<torch::nn::L1Loss>>(m, "L1Loss")
        .def(py::init<torch::nn::Reduction>(), py::arg("reduction") = torch::nn::Reduction::Mean)
        .def("__call__", [](torch::nn::L1Loss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    // Functional API - simplified with inline implementations
    py::module functional = m.def_submodule("functional", "Functional API");

    functional.def("relu", [](const at::Tensor& input, bool inplace) {
        (void)inplace;  // In-place not implemented
        return input.relu();
    }, py::arg("input"), py::arg("inplace") = false);

    functional.def("sigmoid", [](const at::Tensor& input) {
        return input.sigmoid();
    });

    functional.def("tanh", [](const at::Tensor& input) {
        return input.tanh();
    });

    functional.def("softmax", [](const at::Tensor& input, int64_t dim) {
        // CPU softmax implementation
        int64_t ndim = input.dim();
        if (dim < 0) dim += ndim;
        auto max_result = input.max(dim, true);
        at::Tensor max_vals = std::get<0>(max_result);
        at::Tensor shifted = input - max_vals;
        at::Tensor exp_vals = shifted.exp();
        at::Tensor sum_exp = exp_vals.sum(dim, true);
        return exp_vals / sum_exp;
    }, py::arg("input"), py::arg("dim"));

    functional.def("log_softmax", [](const at::Tensor& input, int64_t dim) {
        // log_softmax = input - log(sum(exp(input)))
        int64_t ndim = input.dim();
        if (dim < 0) dim += ndim;
        auto max_result = input.max(dim, true);
        at::Tensor max_vals = std::get<0>(max_result);
        at::Tensor shifted = input - max_vals;
        at::Tensor exp_vals = shifted.exp();
        at::Tensor sum_exp = exp_vals.sum(dim, true);
        return shifted - sum_exp.log();
    }, py::arg("input"), py::arg("dim"));

    functional.def("dropout", [](const at::Tensor& input, double p, bool training, bool inplace) {
        (void)inplace;
        if (!training || p == 0.0) {
            return input;
        }
        // Simple dropout: mask and scale
        at::Tensor mask = torch::rand(input.sizes());
        mask = mask.gt(at::Scalar(p)).to(input.dtype());
        return input * mask / (1.0 - p);
    }, py::arg("input"), py::arg("p") = 0.5, py::arg("training") = true, py::arg("inplace") = false);

    functional.def("linear", [](const at::Tensor& input, const at::Tensor& weight, py::object bias) {
        at::Tensor output;
        at::Tensor weight_t = weight.t();
        if (input.dim() == 1) {
            output = weight.mv(input);
        } else {
            output = input.matmul(weight_t);
        }
        if (!bias.is_none()) {
            output = output + bias.cast<at::Tensor>();
        }
        return output;
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());

    functional.def("mse_loss", [](const at::Tensor& input, const at::Tensor& target, const std::string& reduction) {
        at::Tensor diff = input - target;
        at::Tensor loss = diff * diff;
        if (reduction == "mean") {
            return loss.mean();
        } else if (reduction == "sum") {
            return loss.sum();
        }
        return loss;
    }, py::arg("input"), py::arg("target"), py::arg("reduction") = "mean");

    functional.def("l1_loss", [](const at::Tensor& input, const at::Tensor& target, const std::string& reduction) {
        at::Tensor diff = (input - target).abs();
        if (reduction == "mean") {
            return diff.mean();
        } else if (reduction == "sum") {
            return diff.sum();
        }
        return diff;
    }, py::arg("input"), py::arg("target"), py::arg("reduction") = "mean");

    // Initialization functions - simplified
    m.def("zeros_", [](at::Tensor& tensor) {
        tensor.zero_();
        return tensor;
    });

    m.def("ones_", [](at::Tensor& tensor) {
        tensor.fill_(at::Scalar(1.0));
        return tensor;
    });

    m.def("uniform_", [](at::Tensor& tensor, double a, double b) {
        float* data = tensor.mutable_data_ptr<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            double r = static_cast<double>(::rand()) / RAND_MAX;
            data[i] = static_cast<float>(a + r * (b - a));
        }
        return tensor;
    }, py::arg("tensor"), py::arg("a") = 0.0, py::arg("b") = 1.0);

    m.def("normal_", [](at::Tensor& tensor, double mean, double std_val) {
        float* data = tensor.mutable_data_ptr<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            // Box-Muller transform
            double u1 = static_cast<double>(::rand() + 1) / (RAND_MAX + 1);
            double u2 = static_cast<double>(::rand()) / RAND_MAX;
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2);
            data[i] = static_cast<float>(mean + std_val * z);
        }
        return tensor;
    }, py::arg("tensor"), py::arg("mean") = 0.0, py::arg("std") = 1.0);

    m.def("xavier_uniform_", [](at::Tensor& tensor, double gain) {
        auto size = tensor.sizes();
        int64_t fan_in = size.size() > 1 ? size[1] : size[0];
        int64_t fan_out = size[0];
        double std_val = gain * std::sqrt(2.0 / static_cast<double>(fan_in + fan_out));
        double a = std::sqrt(3.0) * std_val;
        float* data = tensor.mutable_data_ptr<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            double r = static_cast<double>(::rand()) / RAND_MAX;
            data[i] = static_cast<float>(-a + r * 2.0 * a);
        }
        return tensor;
    }, py::arg("tensor"), py::arg("gain") = 1.0);

    m.def("kaiming_uniform_", [](at::Tensor& tensor, double a, const std::string& mode, const std::string& nonlinearity) {
        (void)nonlinearity;
        auto size = tensor.sizes();
        int64_t fan = (mode == "fan_out") ? size[0] : (size.size() > 1 ? size[1] : size[0]);
        double std_val = std::sqrt(2.0 / ((1.0 + a * a) * static_cast<double>(fan)));
        double bound = std::sqrt(3.0) * std_val;
        float* data = tensor.mutable_data_ptr<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            double r = static_cast<double>(::rand()) / RAND_MAX;
            data[i] = static_cast<float>(-bound + r * 2.0 * bound);
        }
        return tensor;
    }, py::arg("tensor"), py::arg("a") = 0.0, py::arg("mode") = "fan_in", py::arg("nonlinearity") = "leaky_relu");
}
