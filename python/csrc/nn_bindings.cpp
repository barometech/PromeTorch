// ============================================================================
// PromeTorch Python Bindings - Neural Network (Full)
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <sstream>
#include <cmath>

#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"
#include "torch/serialization.h"

namespace py = pybind11;

// ============================================================================
// NN Module Bindings (Full)
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
            for (auto* p : self.parameters()) {
                params.push_back(p->data());
            }
            return params;
        })
        .def("named_parameters", [](torch::nn::Module& self) {
            std::vector<std::pair<std::string, at::Tensor>> params;
            for (auto& np : self.named_parameters()) {
                params.push_back(std::make_pair(np.first, np.second->data()));
            }
            return params;
        })
        .def("name", &torch::nn::Module::name)
        .def("is_training", &torch::nn::Module::is_training)
        .def("to", [](torch::nn::Module& self, const std::string& device_str) {
            c10::Device device(device_str);
            self.to(device);
        }, py::arg("device"))
        .def("to", [](torch::nn::Module& self, const c10::Device& device) {
            self.to(device);
        }, py::arg("device"))
        .def("to", [](torch::nn::Module& self, c10::ScalarType dtype) {
            self.to(dtype);
        }, py::arg("dtype"))
        .def("cuda", [](torch::nn::Module& self, int device_index) {
            self.to(c10::Device(c10::DeviceType::CUDA, device_index));
        }, py::arg("device") = 0)
        .def("cpu", [](torch::nn::Module& self) {
            self.to(c10::Device(c10::DeviceType::CPU));
        })
        .def("state_dict", [](torch::nn::Module& self) {
            return self.state_dict();
        })
        .def("load_state_dict", [](torch::nn::Module& self,
                const std::unordered_map<std::string, at::Tensor>& state_dict, bool strict) {
            self.load_state_dict(state_dict, strict);
        }, py::arg("state_dict"), py::arg("strict") = true)
        .def("__repr__", [](torch::nn::Module& self) {
            return torch::nn::module_repr(self);
        })
        .def("children", [](torch::nn::Module& self) {
            std::vector<std::shared_ptr<torch::nn::Module>> result;
            for (auto& nc : self.named_children()) {
                result.push_back(nc.second);
            }
            return result;
        })
        .def("named_children", [](torch::nn::Module& self) {
            return self.named_children();
        });

    // ========================================================================
    // Sequential container
    // ========================================================================
    py::class_<torch::nn::Sequential, torch::nn::Module, std::shared_ptr<torch::nn::Sequential>>(m, "Sequential")
        .def(py::init<>())
        .def(py::init([](py::args modules) {
            auto seq = std::make_shared<torch::nn::Sequential>();
            for (auto& m : modules) {
                auto mod = m.cast<std::shared_ptr<torch::nn::Module>>();
                seq->add(mod);
            }
            return seq;
        }))
        .def("add", [](torch::nn::Sequential& self, std::shared_ptr<torch::nn::Module> module) {
            self.add(module);
        })
        .def("forward", &torch::nn::Sequential::forward)
        .def("__call__", &torch::nn::Sequential::forward)
        .def("__len__", &torch::nn::Sequential::size)
        .def("__getitem__", [](torch::nn::Sequential& self, size_t index) {
            return self[index];
        });

    // ========================================================================
    // ModuleList container
    // ========================================================================
    py::class_<torch::nn::ModuleList, torch::nn::Module, std::shared_ptr<torch::nn::ModuleList>>(m, "ModuleList")
        .def(py::init<>())
        .def(py::init([](py::list modules) {
            auto ml = std::make_shared<torch::nn::ModuleList>();
            for (auto& m : modules) {
                auto mod = m.cast<std::shared_ptr<torch::nn::Module>>();
                ml->append(mod);
            }
            return ml;
        }))
        .def("append", [](torch::nn::ModuleList& self, std::shared_ptr<torch::nn::Module> module) {
            self.append(module);
        })
        .def("__len__", &torch::nn::ModuleList::size)
        .def("__getitem__", [](torch::nn::ModuleList& self, size_t index) {
            return self[index];
        });

    // ========================================================================
    // Linear layer
    // ========================================================================
    py::class_<torch::nn::Linear, torch::nn::Module, std::shared_ptr<torch::nn::Linear>>(m, "Linear")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_features"), py::arg("out_features"), py::arg("bias") = true)
        .def("forward", &torch::nn::Linear::forward)
        .def("__call__", &torch::nn::Linear::forward)
        .def_property_readonly("in_features", &torch::nn::Linear::in_features)
        .def_property_readonly("out_features", &torch::nn::Linear::out_features)
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

    // ========================================================================
    // Activation functions as modules
    // ========================================================================
    py::class_<torch::nn::ReLU, torch::nn::Module, std::shared_ptr<torch::nn::ReLU>>(m, "ReLU")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def("forward", &torch::nn::ReLU::forward)
        .def("__call__", &torch::nn::ReLU::forward);

    py::class_<torch::nn::ReLU6, torch::nn::Module, std::shared_ptr<torch::nn::ReLU6>>(m, "ReLU6")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def("forward", &torch::nn::ReLU6::forward)
        .def("__call__", &torch::nn::ReLU6::forward);

    py::class_<torch::nn::LeakyReLU, torch::nn::Module, std::shared_ptr<torch::nn::LeakyReLU>>(m, "LeakyReLU")
        .def(py::init<double, bool>(), py::arg("negative_slope") = 0.01, py::arg("inplace") = false)
        .def("forward", &torch::nn::LeakyReLU::forward)
        .def("__call__", &torch::nn::LeakyReLU::forward);

    py::class_<torch::nn::PReLU, torch::nn::Module, std::shared_ptr<torch::nn::PReLU>>(m, "PReLU")
        .def(py::init<int64_t, double>(), py::arg("num_parameters") = 1, py::arg("init") = 0.25)
        .def("forward", &torch::nn::PReLU::forward)
        .def("__call__", &torch::nn::PReLU::forward);

    py::class_<torch::nn::ELU, torch::nn::Module, std::shared_ptr<torch::nn::ELU>>(m, "ELU")
        .def(py::init<double, bool>(), py::arg("alpha") = 1.0, py::arg("inplace") = false)
        .def("forward", &torch::nn::ELU::forward)
        .def("__call__", &torch::nn::ELU::forward);

    py::class_<torch::nn::SELU, torch::nn::Module, std::shared_ptr<torch::nn::SELU>>(m, "SELU")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def("forward", &torch::nn::SELU::forward)
        .def("__call__", &torch::nn::SELU::forward);

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

    py::class_<torch::nn::LogSoftmax, torch::nn::Module, std::shared_ptr<torch::nn::LogSoftmax>>(m, "LogSoftmax")
        .def(py::init<int64_t>(), py::arg("dim") = -1)
        .def("forward", &torch::nn::LogSoftmax::forward)
        .def("__call__", &torch::nn::LogSoftmax::forward);

    py::class_<torch::nn::GELU, torch::nn::Module, std::shared_ptr<torch::nn::GELU>>(m, "GELU")
        .def(py::init<std::string>(), py::arg("approximate") = "none")
        .def("forward", &torch::nn::GELU::forward)
        .def("__call__", &torch::nn::GELU::forward);

    py::class_<torch::nn::SiLU, torch::nn::Module, std::shared_ptr<torch::nn::SiLU>>(m, "SiLU")
        .def(py::init<>())
        .def("forward", &torch::nn::SiLU::forward)
        .def("__call__", &torch::nn::SiLU::forward);

    py::class_<torch::nn::Mish, torch::nn::Module, std::shared_ptr<torch::nn::Mish>>(m, "Mish")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def("forward", &torch::nn::Mish::forward)
        .def("__call__", &torch::nn::Mish::forward);

    py::class_<torch::nn::Softplus, torch::nn::Module, std::shared_ptr<torch::nn::Softplus>>(m, "Softplus")
        .def(py::init<double, double>(), py::arg("beta") = 1.0, py::arg("threshold") = 20.0)
        .def("forward", &torch::nn::Softplus::forward)
        .def("__call__", &torch::nn::Softplus::forward);

    py::class_<torch::nn::Softsign, torch::nn::Module, std::shared_ptr<torch::nn::Softsign>>(m, "Softsign")
        .def(py::init<>())
        .def("forward", &torch::nn::Softsign::forward)
        .def("__call__", &torch::nn::Softsign::forward);

    py::class_<torch::nn::Hardtanh, torch::nn::Module, std::shared_ptr<torch::nn::Hardtanh>>(m, "Hardtanh")
        .def(py::init<double, double, bool>(),
             py::arg("min_val") = -1.0, py::arg("max_val") = 1.0, py::arg("inplace") = false)
        .def("forward", &torch::nn::Hardtanh::forward)
        .def("__call__", &torch::nn::Hardtanh::forward);

    py::class_<torch::nn::Hardsigmoid, torch::nn::Module, std::shared_ptr<torch::nn::Hardsigmoid>>(m, "Hardsigmoid")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def("forward", &torch::nn::Hardsigmoid::forward)
        .def("__call__", &torch::nn::Hardsigmoid::forward);

    py::class_<torch::nn::Hardswish, torch::nn::Module, std::shared_ptr<torch::nn::Hardswish>>(m, "Hardswish")
        .def(py::init<bool>(), py::arg("inplace") = false)
        .def("forward", &torch::nn::Hardswish::forward)
        .def("__call__", &torch::nn::Hardswish::forward);

    // ========================================================================
    // Dropout
    // ========================================================================
    py::class_<torch::nn::Dropout, torch::nn::Module, std::shared_ptr<torch::nn::Dropout>>(m, "Dropout")
        .def(py::init<double, bool>(), py::arg("p") = 0.5, py::arg("inplace") = false)
        .def("forward", &torch::nn::Dropout::forward)
        .def("__call__", &torch::nn::Dropout::forward);

    // ========================================================================
    // Normalization
    // ========================================================================
    py::class_<torch::nn::BatchNorm2d, torch::nn::Module, std::shared_ptr<torch::nn::BatchNorm2d>>(m, "BatchNorm2d")
        .def(py::init<int64_t, double, double, bool, bool>(),
             py::arg("num_features"), py::arg("eps") = 1e-5, py::arg("momentum") = 0.1,
             py::arg("affine") = true, py::arg("track_running_stats") = true)
        .def("forward", &torch::nn::BatchNorm2d::forward)
        .def("__call__", &torch::nn::BatchNorm2d::forward);

    py::class_<torch::nn::LayerNorm, torch::nn::Module, std::shared_ptr<torch::nn::LayerNorm>>(m, "LayerNorm")
        .def(py::init<std::vector<int64_t>, double, bool>(),
             py::arg("normalized_shape"), py::arg("eps") = 1e-5, py::arg("elementwise_affine") = true)
        .def("forward", &torch::nn::LayerNorm::forward)
        .def("__call__", &torch::nn::LayerNorm::forward);

    // ========================================================================
    // Convolution
    // ========================================================================
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

    // ========================================================================
    // Pooling
    // ========================================================================
    py::class_<torch::nn::MaxPool2d, torch::nn::Module, std::shared_ptr<torch::nn::MaxPool2d>>(m, "MaxPool2d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"), py::arg("stride") = 0, py::arg("padding") = 0)
        .def("forward", &torch::nn::MaxPool2d::forward)
        .def("__call__", &torch::nn::MaxPool2d::forward);

    py::class_<torch::nn::AvgPool2d, torch::nn::Module, std::shared_ptr<torch::nn::AvgPool2d>>(m, "AvgPool2d")
        .def(py::init<int64_t, int64_t, int64_t>(),
             py::arg("kernel_size"), py::arg("stride") = 0, py::arg("padding") = 0)
        .def("forward", &torch::nn::AvgPool2d::forward)
        .def("__call__", &torch::nn::AvgPool2d::forward);

    // ========================================================================
    // Embedding
    // ========================================================================
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

    // ========================================================================
    // Recurrent modules
    // ========================================================================
    py::class_<torch::nn::RNN, torch::nn::Module, std::shared_ptr<torch::nn::RNN>>(m, "RNN")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, double, bool>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("num_layers") = 1, py::arg("bias") = true,
             py::arg("batch_first") = false, py::arg("dropout") = 0.0,
             py::arg("bidirectional") = false)
        .def("forward", [](torch::nn::RNN& self, const at::Tensor& input, py::object h0) {
            if (h0.is_none()) {
                return self.forward_rnn(input);
            }
            return self.forward_rnn(input, h0.cast<at::Tensor>());
        }, py::arg("input"), py::arg("h0") = py::none())
        .def("__call__", [](torch::nn::RNN& self, const at::Tensor& input, py::object h0) {
            if (h0.is_none()) {
                return self.forward_rnn(input);
            }
            return self.forward_rnn(input, h0.cast<at::Tensor>());
        }, py::arg("input"), py::arg("h0") = py::none());

    py::class_<torch::nn::LSTM::LSTMOutput>(m, "LSTMOutput")
        .def_readonly("output", &torch::nn::LSTM::LSTMOutput::output)
        .def_readonly("h_n", &torch::nn::LSTM::LSTMOutput::h_n)
        .def_readonly("c_n", &torch::nn::LSTM::LSTMOutput::c_n);

    py::class_<torch::nn::LSTM, torch::nn::Module, std::shared_ptr<torch::nn::LSTM>>(m, "LSTM")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, double, bool>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("num_layers") = 1, py::arg("bias") = true,
             py::arg("batch_first") = false, py::arg("dropout") = 0.0,
             py::arg("bidirectional") = false)
        .def("forward", [](torch::nn::LSTM& self, const at::Tensor& input,
                           py::object h0, py::object c0) {
            at::Tensor h = h0.is_none() ? at::Tensor() : h0.cast<at::Tensor>();
            at::Tensor c = c0.is_none() ? at::Tensor() : c0.cast<at::Tensor>();
            auto result = self.forward_lstm(input, h, c);
            return py::make_tuple(result.output, result.h_n, result.c_n);
        }, py::arg("input"), py::arg("h0") = py::none(), py::arg("c0") = py::none())
        .def("__call__", [](torch::nn::LSTM& self, const at::Tensor& input,
                            py::object h0, py::object c0) {
            at::Tensor h = h0.is_none() ? at::Tensor() : h0.cast<at::Tensor>();
            at::Tensor c = c0.is_none() ? at::Tensor() : c0.cast<at::Tensor>();
            auto result = self.forward_lstm(input, h, c);
            return py::make_tuple(result.output, result.h_n, result.c_n);
        }, py::arg("input"), py::arg("h0") = py::none(), py::arg("c0") = py::none());

    py::class_<torch::nn::GRU, torch::nn::Module, std::shared_ptr<torch::nn::GRU>>(m, "GRU")
        .def(py::init<int64_t, int64_t, int64_t, bool, bool, double, bool>(),
             py::arg("input_size"), py::arg("hidden_size"),
             py::arg("num_layers") = 1, py::arg("bias") = true,
             py::arg("batch_first") = false, py::arg("dropout") = 0.0,
             py::arg("bidirectional") = false)
        .def("forward", [](torch::nn::GRU& self, const at::Tensor& input, py::object h0) {
            if (h0.is_none()) {
                return self.forward_gru(input);
            }
            return self.forward_gru(input, h0.cast<at::Tensor>());
        }, py::arg("input"), py::arg("h0") = py::none())
        .def("__call__", [](torch::nn::GRU& self, const at::Tensor& input, py::object h0) {
            if (h0.is_none()) {
                return self.forward_gru(input);
            }
            return self.forward_gru(input, h0.cast<at::Tensor>());
        }, py::arg("input"), py::arg("h0") = py::none());

    // ========================================================================
    // Reduction enum
    // ========================================================================
    py::enum_<torch::nn::Reduction>(m, "Reduction")
        .value("None_", torch::nn::Reduction::None)
        .value("Mean", torch::nn::Reduction::Mean)
        .value("Sum", torch::nn::Reduction::Sum)
        .export_values();

    // ========================================================================
    // Loss functions
    // ========================================================================
    py::class_<torch::nn::MSELoss, torch::nn::Module, std::shared_ptr<torch::nn::MSELoss>>(m, "MSELoss")
        .def(py::init<torch::nn::Reduction>(), py::arg("reduction") = torch::nn::Reduction::Mean)
        .def("forward", [](torch::nn::MSELoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        })
        .def("__call__", [](torch::nn::MSELoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    py::class_<torch::nn::CrossEntropyLoss, torch::nn::Module, std::shared_ptr<torch::nn::CrossEntropyLoss>>(m, "CrossEntropyLoss")
        .def(py::init<torch::nn::Reduction, int64_t, double>(),
             py::arg("reduction") = torch::nn::Reduction::Mean,
             py::arg("ignore_index") = -100, py::arg("label_smoothing") = 0.0)
        .def("forward", [](torch::nn::CrossEntropyLoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        })
        .def("__call__", [](torch::nn::CrossEntropyLoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    py::class_<torch::nn::NLLLoss, torch::nn::Module, std::shared_ptr<torch::nn::NLLLoss>>(m, "NLLLoss")
        .def(py::init<torch::nn::Reduction, int64_t>(),
             py::arg("reduction") = torch::nn::Reduction::Mean, py::arg("ignore_index") = -100)
        .def("forward", [](torch::nn::NLLLoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        })
        .def("__call__", [](torch::nn::NLLLoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    py::class_<torch::nn::BCELoss, torch::nn::Module, std::shared_ptr<torch::nn::BCELoss>>(m, "BCELoss")
        .def(py::init<torch::nn::Reduction>(), py::arg("reduction") = torch::nn::Reduction::Mean)
        .def("forward", [](torch::nn::BCELoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        })
        .def("__call__", [](torch::nn::BCELoss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    py::class_<torch::nn::L1Loss, torch::nn::Module, std::shared_ptr<torch::nn::L1Loss>>(m, "L1Loss")
        .def(py::init<torch::nn::Reduction>(), py::arg("reduction") = torch::nn::Reduction::Mean)
        .def("forward", [](torch::nn::L1Loss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        })
        .def("__call__", [](torch::nn::L1Loss& self, const at::Tensor& input, const at::Tensor& target) {
            return self.forward(input, target);
        });

    // ========================================================================
    // Functional API
    // ========================================================================
    py::module functional = m.def_submodule("functional", "Functional API");

    functional.def("relu", [](const at::Tensor& input, bool inplace) {
        (void)inplace;
        return input.relu();
    }, py::arg("input"), py::arg("inplace") = false);

    functional.def("leaky_relu", [](const at::Tensor& input, double negative_slope, bool inplace) {
        (void)inplace;
        // leaky_relu: max(0, x) + negative_slope * min(0, x)
        // Avoid Bool->Float conversion
        at::Tensor result = input.clone();
        float ns = static_cast<float>(negative_slope);
        float* d = result.mutable_data_ptr<float>();
        for (int64_t i = 0; i < result.numel(); i++) {
            if (d[i] < 0.0f) d[i] *= ns;
        }
        return result;
    }, py::arg("input"), py::arg("negative_slope") = 0.01, py::arg("inplace") = false);

    functional.def("elu", [](const at::Tensor& input, double alpha, bool inplace) {
        (void)inplace;
        at::Tensor result = input.clone();
        float a = static_cast<float>(alpha);
        float* d = result.mutable_data_ptr<float>();
        for (int64_t i = 0; i < result.numel(); i++) {
            if (d[i] <= 0.0f) d[i] = a * (std::exp(d[i]) - 1.0f);
        }
        return result;
    }, py::arg("input"), py::arg("alpha") = 1.0, py::arg("inplace") = false);

    functional.def("selu", [](const at::Tensor& input, bool inplace) {
        (void)inplace;
        constexpr float alpha = 1.6732632423543772f;
        constexpr float scale = 1.0507009873554805f;
        at::Tensor result = input.clone();
        float* d = result.mutable_data_ptr<float>();
        for (int64_t i = 0; i < result.numel(); i++) {
            if (d[i] <= 0.0f)
                d[i] = scale * alpha * (std::exp(d[i]) - 1.0f);
            else
                d[i] = scale * d[i];
        }
        return result;
    }, py::arg("input"), py::arg("inplace") = false);

    functional.def("gelu", [](const at::Tensor& input) {
        // GELU(x) = x * Phi(x) approx x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double sqrt_2_over_pi = std::sqrt(2.0 / 3.14159265358979323846);
        at::Tensor x3 = input * input * input;
        at::Tensor inner = (input + x3 * 0.044715) * sqrt_2_over_pi;
        return input * (inner.tanh() + 1.0) * 0.5;
    }, py::arg("input"));

    functional.def("silu", [](const at::Tensor& input) {
        return input * input.sigmoid();
    }, py::arg("input"));

    functional.def("sigmoid", [](const at::Tensor& input) {
        return input.sigmoid();
    });

    functional.def("tanh", [](const at::Tensor& input) {
        return input.tanh();
    });

    functional.def("softmax", [](const at::Tensor& input, int64_t dim) {
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
        // Generate random mask: 1.0 where rand > p, 0.0 otherwise
        // Avoid Bool->Float conversion by comparing floats directly
        at::Tensor rand_vals = torch::rand(input.sizes());
        // Create float mask: rand > p gives 1.0, else 0.0
        // Use element-wise: (rand - p) > 0 ? 1 : 0 via relu + sign-like
        float threshold = static_cast<float>(p);
        float* rv = rand_vals.mutable_data_ptr<float>();
        for (int64_t i = 0; i < rand_vals.numel(); i++) {
            rv[i] = rv[i] > threshold ? 1.0f : 0.0f;
        }
        return input * rand_vals / (1.0 - p);
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

    functional.def("cross_entropy", [](const at::Tensor& input, const at::Tensor& target,
                                        py::object weight, int64_t ignore_index,
                                        const std::string& reduction, double label_smoothing) {
        // Use CrossEntropyLoss module internally
        torch::nn::Reduction red = torch::nn::Reduction::Mean;
        if (reduction == "sum") red = torch::nn::Reduction::Sum;
        else if (reduction == "none") red = torch::nn::Reduction::None;
        torch::nn::CrossEntropyLoss loss_fn(red, ignore_index, label_smoothing);
        return loss_fn.forward(input, target);
    }, py::arg("input"), py::arg("target"), py::arg("weight") = py::none(),
       py::arg("ignore_index") = -100, py::arg("reduction") = "mean",
       py::arg("label_smoothing") = 0.0);

    functional.def("nll_loss", [](const at::Tensor& input, const at::Tensor& target,
                                   py::object weight, int64_t ignore_index,
                                   const std::string& reduction) {
        torch::nn::Reduction red = torch::nn::Reduction::Mean;
        if (reduction == "sum") red = torch::nn::Reduction::Sum;
        else if (reduction == "none") red = torch::nn::Reduction::None;
        torch::nn::NLLLoss loss_fn(red, ignore_index);
        return loss_fn.forward(input, target);
    }, py::arg("input"), py::arg("target"), py::arg("weight") = py::none(),
       py::arg("ignore_index") = -100, py::arg("reduction") = "mean");

    functional.def("binary_cross_entropy", [](const at::Tensor& input, const at::Tensor& target,
                                               py::object weight, const std::string& reduction) {
        torch::nn::Reduction red = torch::nn::Reduction::Mean;
        if (reduction == "sum") red = torch::nn::Reduction::Sum;
        else if (reduction == "none") red = torch::nn::Reduction::None;
        torch::nn::BCELoss loss_fn(red);
        return loss_fn.forward(input, target);
    }, py::arg("input"), py::arg("target"), py::arg("weight") = py::none(),
       py::arg("reduction") = "mean");

    functional.def("pad", [](const at::Tensor& input, std::vector<int64_t> pad, const std::string& mode, double value) {
        // Constant padding for N-d tensors
        // pad format: (left, right) for 1D last dim, (left, right, top, bottom) for 2D last 2 dims
        at::Tensor result = input;

        int64_t ndim = input.dim();
        int64_t num_pad_dims = static_cast<int64_t>(pad.size()) / 2;

        // Build output shape
        std::vector<int64_t> out_shape(input.sizes().begin(), input.sizes().end());
        for (int64_t i = 0; i < num_pad_dims; ++i) {
            int64_t dim = ndim - 1 - i;  // pad starts from last dim
            int64_t pad_before = pad[2 * i];
            int64_t pad_after = pad[2 * i + 1];
            out_shape[dim] += pad_before + pad_after;
        }

        at::Tensor output = torch::full(out_shape, value, at::TensorOptions().dtype(input.dtype()));

        // Copy input into the right position
        // Build slice indices
        std::vector<std::pair<int64_t, int64_t>> slices;
        for (int64_t d = 0; d < ndim; ++d) {
            slices.push_back({0, input.size(d)});
        }
        for (int64_t i = 0; i < num_pad_dims; ++i) {
            int64_t dim = ndim - 1 - i;
            int64_t pad_before = pad[2 * i];
            slices[dim] = {pad_before, pad_before + input.size(dim)};
        }

        // Use slice operations to copy
        at::Tensor dst = output;
        at::Tensor src = input;
        for (int64_t d = 0; d < ndim; ++d) {
            dst = dst.slice(d, slices[d].first, slices[d].second);
        }
        dst.copy_(src);

        return output;
    }, py::arg("input"), py::arg("pad"), py::arg("mode") = "constant", py::arg("value") = 0.0);

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

    // ========================================================================
    // nn.utils: gradient clipping
    // ========================================================================
    py::module utils = m.def_submodule("utils", "NN utilities");

    utils.def("clip_grad_norm_", [](py::list params, double max_norm, double norm_type) {
        // Collect parameter tensors, compute total norm, then clip
        double total_norm = 0.0;
        std::vector<at::Tensor> grads;
        for (auto& t : params) {
            at::Tensor tensor = t.cast<at::Tensor>();
            // Parameters from Python are the data tensors; we need to check grad
            // In our framework, grad is stored on the tensor
            at::Tensor g = tensor.grad();
            if (g.defined()) {
                grads.push_back(g);
                const float* gd = g.data_ptr<float>();
                for (int64_t i = 0; i < g.numel(); i++) {
                    total_norm += (double)gd[i] * gd[i];
                }
            }
        }
        total_norm = std::sqrt(total_norm);
        double clip_coef = max_norm / (total_norm + 1e-6);
        if (clip_coef < 1.0) {
            for (auto& g : grads) {
                g.mul_(at::Scalar(clip_coef));
            }
        }
        return total_norm;
    }, py::arg("parameters"), py::arg("max_norm"), py::arg("norm_type") = 2.0);

    // ========================================================================
    // Initialization functions
    // ========================================================================
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

    // orthogonal_ initialization (via QR decomposition approximation)
    m.def("orthogonal_", [](at::Tensor& tensor, double gain) {
        // Simple orthogonal init: fill with random normal, then orthogonalize via Gram-Schmidt
        int64_t rows = tensor.size(0);
        int64_t cols = tensor.numel() / rows;
        float* data = tensor.mutable_data_ptr<float>();

        // Fill with random normal
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            double u1 = (double)(::rand() + 1) / (RAND_MAX + 1.0);
            double u2 = (double)::rand() / RAND_MAX;
            data[i] = (float)(std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979323846 * u2));
        }

        // Gram-Schmidt orthogonalization (for rows)
        int64_t n = std::min(rows, cols);
        for (int64_t i = 0; i < n; ++i) {
            float* row_i = data + i * cols;
            // Subtract projections of previous rows
            for (int64_t j = 0; j < i; ++j) {
                float* row_j = data + j * cols;
                // dot product
                double dot = 0.0;
                for (int64_t k = 0; k < cols; ++k) {
                    dot += (double)row_i[k] * row_j[k];
                }
                for (int64_t k = 0; k < cols; ++k) {
                    row_i[k] -= (float)(dot * row_j[k]);
                }
            }
            // Normalize
            double norm = 0.0;
            for (int64_t k = 0; k < cols; ++k) {
                norm += (double)row_i[k] * row_i[k];
            }
            norm = std::sqrt(norm);
            if (norm > 1e-8) {
                float scale = (float)(gain / norm);
                for (int64_t k = 0; k < cols; ++k) {
                    row_i[k] *= scale;
                }
            }
        }

        // Scale remaining rows
        for (int64_t i = n; i < rows; ++i) {
            float* row_i = data + i * cols;
            double norm = 0.0;
            for (int64_t k = 0; k < cols; ++k) norm += (double)row_i[k] * row_i[k];
            norm = std::sqrt(norm);
            if (norm > 1e-8) {
                float scale = (float)(gain / norm);
                for (int64_t k = 0; k < cols; ++k) row_i[k] *= scale;
            }
        }

        return tensor;
    }, py::arg("tensor"), py::arg("gain") = 1.0);
}
