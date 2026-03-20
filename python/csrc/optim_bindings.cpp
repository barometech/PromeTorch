// ============================================================================
// PromeTorch Python Bindings - Optimizers
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"
#include "torch/optim/optim.h"

namespace py = pybind11;

// ============================================================================
// Per-optimizer parameter storage
// ============================================================================
// Instead of a global static vector that leaks forever, each optimizer gets
// its own shared container of Parameters. The container is co-owned by the
// shared_ptr<Optimizer> returned to Python, so it is destroyed when the
// optimizer is garbage-collected.

using ParamStorage = std::vector<std::unique_ptr<torch::nn::Parameter>>;

struct ParamsAndPtrs {
    std::shared_ptr<ParamStorage> storage;
    std::vector<torch::nn::Parameter*> ptrs;
};

ParamsAndPtrs params_from_tensors(py::list tensors) {
    ParamsAndPtrs result;
    result.storage = std::make_shared<ParamStorage>();
    result.storage->reserve(tensors.size());
    for (auto& t : tensors) {
        auto tensor = t.cast<at::Tensor>();
        auto param = std::make_unique<torch::nn::Parameter>(tensor, true);
        result.ptrs.push_back(param.get());
        result.storage->push_back(std::move(param));
    }
    return result;
}

// Helper: wrap an optimizer shared_ptr so that it co-owns the param storage.
// When Python releases the optimizer, both the optimizer and param storage
// are freed (in that order, since the custom deleter releases optimizer first).
template<typename OptimizerT>
std::shared_ptr<OptimizerT> make_optimizer_with_params(
    std::shared_ptr<OptimizerT> optimizer,
    std::shared_ptr<ParamStorage> param_storage)
{
    // Create an aliasing shared_ptr: points to the same optimizer object,
    // but the control block now also prevents param_storage from being freed.
    // We capture param_storage in the deleter.
    auto* raw = optimizer.get();
    return std::shared_ptr<OptimizerT>(raw,
        [opt = std::move(optimizer), params = std::move(param_storage)](OptimizerT*) mutable {
            // Destroy optimizer first (it references params), then params
            opt.reset();
            params.reset();
        });
}

// ============================================================================
// Optimizer Bindings
// ============================================================================

void init_optim_bindings(py::module& m) {

    // Optimizer base class
    py::class_<torch::optim::Optimizer, std::shared_ptr<torch::optim::Optimizer>>(m, "Optimizer")
        .def("step", &torch::optim::Optimizer::step)
        .def("zero_grad", [](torch::optim::Optimizer& self, bool set_to_none) {
            self.zero_grad(set_to_none);
        }, py::arg("set_to_none") = false);

    // SGDOptions
    py::class_<torch::optim::SGDOptions>(m, "SGDOptions")
        .def(py::init<double>(), py::arg("lr") = 0.01)
        .def_readwrite("lr", &torch::optim::SGDOptions::lr)
        .def_readwrite("momentum", &torch::optim::SGDOptions::momentum)
        .def_readwrite("dampening", &torch::optim::SGDOptions::dampening)
        .def_readwrite("weight_decay", &torch::optim::SGDOptions::weight_decay)
        .def_readwrite("nesterov", &torch::optim::SGDOptions::nesterov);

    // SGD
    py::class_<torch::optim::SGD, torch::optim::Optimizer, std::shared_ptr<torch::optim::SGD>>(m, "SGD")
        .def(py::init([](py::list params, double lr, double momentum, double dampening,
                        double weight_decay, bool nesterov) {
            auto pp = params_from_tensors(params);
            torch::optim::SGDOptions opts(lr);
            opts.momentum = momentum;
            opts.dampening = dampening;
            opts.weight_decay = weight_decay;
            opts.nesterov = nesterov;
            auto opt = std::make_shared<torch::optim::SGD>(pp.ptrs, opts);
            return make_optimizer_with_params(std::move(opt), std::move(pp.storage));
        }), py::arg("params"), py::arg("lr") = 0.01, py::arg("momentum") = 0.0,
           py::arg("dampening") = 0.0, py::arg("weight_decay") = 0.0, py::arg("nesterov") = false)
        .def("step", &torch::optim::SGD::step)
        .def("zero_grad", [](torch::optim::SGD& self, bool set_to_none) {
            self.zero_grad(set_to_none);
        }, py::arg("set_to_none") = false);

    // AdamOptions
    py::class_<torch::optim::AdamOptions>(m, "AdamOptions")
        .def(py::init<double>(), py::arg("lr") = 0.001)
        .def_readwrite("lr", &torch::optim::AdamOptions::lr)
        .def_readwrite("beta1", &torch::optim::AdamOptions::beta1)
        .def_readwrite("beta2", &torch::optim::AdamOptions::beta2)
        .def_readwrite("eps", &torch::optim::AdamOptions::eps)
        .def_readwrite("weight_decay", &torch::optim::AdamOptions::weight_decay)
        .def_readwrite("amsgrad", &torch::optim::AdamOptions::amsgrad);

    // Adam
    py::class_<torch::optim::Adam, torch::optim::Optimizer, std::shared_ptr<torch::optim::Adam>>(m, "Adam")
        .def(py::init([](py::list params, double lr, py::tuple betas,
                        double eps, double weight_decay, bool amsgrad) {
            double beta1 = betas[0].cast<double>();
            double beta2 = betas[1].cast<double>();
            auto pp = params_from_tensors(params);
            torch::optim::AdamOptions opts(lr);
            opts.beta1 = beta1;
            opts.beta2 = beta2;
            opts.eps = eps;
            opts.weight_decay = weight_decay;
            opts.amsgrad = amsgrad;
            auto opt = std::make_shared<torch::optim::Adam>(pp.ptrs, opts);
            return make_optimizer_with_params(std::move(opt), std::move(pp.storage));
        }), py::arg("params"), py::arg("lr") = 0.001, py::arg("betas") = py::make_tuple(0.9, 0.999),
           py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0, py::arg("amsgrad") = false)
        .def("step", &torch::optim::Adam::step)
        .def("zero_grad", [](torch::optim::Adam& self, bool set_to_none) {
            self.zero_grad(set_to_none);
        }, py::arg("set_to_none") = false);

    // AdamWOptions
    py::class_<torch::optim::AdamWOptions>(m, "AdamWOptions")
        .def(py::init<double>(), py::arg("lr") = 0.001)
        .def_readwrite("lr", &torch::optim::AdamWOptions::lr)
        .def_readwrite("beta1", &torch::optim::AdamWOptions::beta1)
        .def_readwrite("beta2", &torch::optim::AdamWOptions::beta2)
        .def_readwrite("eps", &torch::optim::AdamWOptions::eps)
        .def_readwrite("weight_decay", &torch::optim::AdamWOptions::weight_decay)
        .def_readwrite("amsgrad", &torch::optim::AdamWOptions::amsgrad);

    // AdamW
    py::class_<torch::optim::AdamW, torch::optim::Optimizer, std::shared_ptr<torch::optim::AdamW>>(m, "AdamW")
        .def(py::init([](py::list params, double lr, py::tuple betas,
                        double eps, double weight_decay, bool amsgrad) {
            double beta1 = betas[0].cast<double>();
            double beta2 = betas[1].cast<double>();
            auto pp = params_from_tensors(params);
            torch::optim::AdamWOptions opts(lr);
            opts.beta1 = beta1;
            opts.beta2 = beta2;
            opts.eps = eps;
            opts.weight_decay = weight_decay;
            opts.amsgrad = amsgrad;
            auto opt = std::make_shared<torch::optim::AdamW>(pp.ptrs, opts);
            return make_optimizer_with_params(std::move(opt), std::move(pp.storage));
        }), py::arg("params"), py::arg("lr") = 0.001, py::arg("betas") = py::make_tuple(0.9, 0.999),
           py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.01, py::arg("amsgrad") = false)
        .def("step", &torch::optim::AdamW::step)
        .def("zero_grad", [](torch::optim::AdamW& self, bool set_to_none) {
            self.zero_grad(set_to_none);
        }, py::arg("set_to_none") = false);

    // RMSpropOptions
    py::class_<torch::optim::RMSpropOptions>(m, "RMSpropOptions")
        .def(py::init<double>(), py::arg("lr") = 0.01)
        .def_readwrite("lr", &torch::optim::RMSpropOptions::lr)
        .def_readwrite("alpha", &torch::optim::RMSpropOptions::alpha)
        .def_readwrite("eps", &torch::optim::RMSpropOptions::eps)
        .def_readwrite("weight_decay", &torch::optim::RMSpropOptions::weight_decay)
        .def_readwrite("momentum", &torch::optim::RMSpropOptions::momentum)
        .def_readwrite("centered", &torch::optim::RMSpropOptions::centered);

    // RMSprop
    py::class_<torch::optim::RMSprop, torch::optim::Optimizer, std::shared_ptr<torch::optim::RMSprop>>(m, "RMSprop")
        .def(py::init([](py::list params, double lr, double alpha, double eps,
                        double weight_decay, double momentum, bool centered) {
            auto pp = params_from_tensors(params);
            torch::optim::RMSpropOptions opts(lr);
            opts.alpha = alpha;
            opts.eps = eps;
            opts.weight_decay = weight_decay;
            opts.momentum = momentum;
            opts.centered = centered;
            auto opt = std::make_shared<torch::optim::RMSprop>(pp.ptrs, opts);
            return make_optimizer_with_params(std::move(opt), std::move(pp.storage));
        }), py::arg("params"), py::arg("lr") = 0.01, py::arg("alpha") = 0.99,
           py::arg("eps") = 1e-8, py::arg("weight_decay") = 0.0,
           py::arg("momentum") = 0.0, py::arg("centered") = false)
        .def("step", &torch::optim::RMSprop::step)
        .def("zero_grad", [](torch::optim::RMSprop& self, bool set_to_none) {
            self.zero_grad(set_to_none);
        }, py::arg("set_to_none") = false);

    // ========================================================================
    // Learning Rate Schedulers
    // ========================================================================

    py::module lr_scheduler = m.def_submodule("lr_scheduler", "Learning rate schedulers");

    // LRScheduler base class
    py::class_<torch::optim::LRScheduler, std::shared_ptr<torch::optim::LRScheduler>>(lr_scheduler, "LRScheduler")
        .def("step", [](torch::optim::LRScheduler& self) { self.step(); })
        .def("get_lr", &torch::optim::LRScheduler::get_lr)
        .def("get_last_lr", &torch::optim::LRScheduler::get_last_lr);

    // StepLR - takes Optimizer& reference
    py::class_<torch::optim::StepLR, torch::optim::LRScheduler, std::shared_ptr<torch::optim::StepLR>>(lr_scheduler, "StepLR")
        .def(py::init([](torch::optim::Optimizer& optimizer, int64_t step_size, double gamma) {
            return std::make_shared<torch::optim::StepLR>(optimizer, step_size, gamma);
        }), py::arg("optimizer"), py::arg("step_size"), py::arg("gamma") = 0.1)
        .def("step", [](torch::optim::StepLR& self) { self.step(); });

    // ExponentialLR
    py::class_<torch::optim::ExponentialLR, torch::optim::LRScheduler, std::shared_ptr<torch::optim::ExponentialLR>>(lr_scheduler, "ExponentialLR")
        .def(py::init([](torch::optim::Optimizer& optimizer, double gamma) {
            return std::make_shared<torch::optim::ExponentialLR>(optimizer, gamma);
        }), py::arg("optimizer"), py::arg("gamma"))
        .def("step", [](torch::optim::ExponentialLR& self) { self.step(); });

    // CosineAnnealingLR
    py::class_<torch::optim::CosineAnnealingLR, torch::optim::LRScheduler, std::shared_ptr<torch::optim::CosineAnnealingLR>>(lr_scheduler, "CosineAnnealingLR")
        .def(py::init([](torch::optim::Optimizer& optimizer, int64_t T_max, double eta_min) {
            return std::make_shared<torch::optim::CosineAnnealingLR>(optimizer, T_max, eta_min);
        }), py::arg("optimizer"), py::arg("T_max"), py::arg("eta_min") = 0.0)
        .def("step", [](torch::optim::CosineAnnealingLR& self) { self.step(); });

    // MultiStepLR
    py::class_<torch::optim::MultiStepLR, torch::optim::LRScheduler, std::shared_ptr<torch::optim::MultiStepLR>>(lr_scheduler, "MultiStepLR")
        .def(py::init([](torch::optim::Optimizer& optimizer, std::vector<int64_t> milestones, double gamma) {
            return std::make_shared<torch::optim::MultiStepLR>(optimizer, milestones, gamma);
        }), py::arg("optimizer"), py::arg("milestones"), py::arg("gamma") = 0.1)
        .def("step", [](torch::optim::MultiStepLR& self) { self.step(); });

    // LinearLR
    py::class_<torch::optim::LinearLR, torch::optim::LRScheduler, std::shared_ptr<torch::optim::LinearLR>>(lr_scheduler, "LinearLR")
        .def(py::init([](torch::optim::Optimizer& optimizer, double start_factor, double end_factor, int64_t total_iters) {
            return std::make_shared<torch::optim::LinearLR>(optimizer, start_factor, end_factor, total_iters);
        }), py::arg("optimizer"), py::arg("start_factor") = 1.0/3.0, py::arg("end_factor") = 1.0, py::arg("total_iters") = 5)
        .def("step", [](torch::optim::LinearLR& self) { self.step(); });

    // ReduceLROnPlateau (not derived from LRScheduler, separate class)
    py::class_<torch::optim::ReduceLROnPlateau>(lr_scheduler, "ReduceLROnPlateau")
        .def(py::init([](torch::optim::Optimizer& optimizer, const std::string& mode, double factor,
                        int64_t patience, double threshold, double min_lr) {
            auto m = (mode == "max") ? torch::optim::ReduceLROnPlateau::Mode::Max
                                     : torch::optim::ReduceLROnPlateau::Mode::Min;
            return torch::optim::ReduceLROnPlateau(optimizer, m, factor, patience, threshold, min_lr);
        }), py::arg("optimizer"), py::arg("mode") = "min", py::arg("factor") = 0.1,
           py::arg("patience") = 10, py::arg("threshold") = 1e-4, py::arg("min_lr") = 0.0)
        .def("step", &torch::optim::ReduceLROnPlateau::step, py::arg("metric"))
        .def("get_last_lr", &torch::optim::ReduceLROnPlateau::get_last_lr);

    // OneCycleLR (also separate class)
    py::class_<torch::optim::OneCycleLR>(lr_scheduler, "OneCycleLR")
        .def(py::init([](torch::optim::Optimizer& optimizer, double max_lr, int64_t total_steps,
                        double pct_start, double div_factor, double final_div_factor) {
            return torch::optim::OneCycleLR(optimizer, max_lr, total_steps, pct_start, div_factor, final_div_factor);
        }), py::arg("optimizer"), py::arg("max_lr"), py::arg("total_steps"),
           py::arg("pct_start") = 0.3, py::arg("div_factor") = 25.0, py::arg("final_div_factor") = 1e4)
        .def("step", &torch::optim::OneCycleLR::step)
        .def("get_last_lr", &torch::optim::OneCycleLR::get_last_lr);
}
