// ============================================================================
// PromeTorch Python Bindings — New Modules
// ----------------------------------------------------------------------------
// Exposes the following C++ subsystems as pybind11 submodules on the
// top-level ``promethorch._C`` extension:
//
//   _C.parallel          torch::nn::parallel (TP, pipeline)
//   _C.distributed       torch::distributed  (DDP, FSDP, launcher)
//   _C.trainer           torch::trainer      (LightningModule, Trainer)
//   _C.onnx_export       torch::onnx
//   _C.mlir_export       torch::mlir
//   _C.mobile_export     torch::mobile
//   _C.jit_compile       torch::jit
//   _C.vision            torch::vision
//   _C.quantization      torch::quantization (QAT)
//   _C.autograd_fwd      torch::autograd::forward_ad / vmap
//   _C.serve             minimal LLM engine scaffold (Python-side filled in)
//
// Each submodule binds the key classes/functions. Python-side wrappers in
// ``promethorch.<module>`` provide higher-level APIs and pure-Python
// fallbacks when the C++ symbols aren't available in the built _C module
// (e.g. while running against an older .pyd).
//
// The file is designed to compile cleanly on both MSVC (Windows) and
// LCC/GCC (Elbrus). All bound headers are header-only, so linkage is
// determined by the extension target (see CMakeLists.txt).
// ============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <memory>
#include <string>
#include <vector>

// ---- Core tensor/module types (already bound elsewhere) ----
#include "aten/src/ATen/ATen.h"
#include "torch/nn/nn.h"
#include "torch/nn/module.h"
#include "torch/optim/optimizer.h"

// ---- New subsystems ----
#include "torch/nn/parallel/tensor_parallel.h"
#include "torch/nn/parallel/pipeline.h"
#include "torch/distributed/distributed.h"
#include "torch/distributed/fsdp.h"
#include "torch/distributed/launcher.h"
#include "torch/trainer/trainer.h"
#include "torch/onnx/export.h"
#include "torch/mlir/export.h"
#include "torch/mobile/executor.h"
#include "torch/vision/vision.h"
#include "torch/quantization/qat.h"
#include "torch/autograd/forward_ad.h"
#include "torch/autograd/vmap.h"

namespace py = pybind11;

// ============================================================================
// Helpers
// ============================================================================

namespace {

// Safely unwrap a Python Module-like into std::shared_ptr<nn::Module>.
// Supports both direct promethorch.nn.Module instances and objects that
// expose a ._module attribute (for LightningModule trampolines, etc.).
std::shared_ptr<torch::nn::Module> to_module_ptr(py::object obj) {
    try {
        return obj.cast<std::shared_ptr<torch::nn::Module>>();
    } catch (const py::cast_error&) {
        if (py::hasattr(obj, "_module")) {
            return py::cast<std::shared_ptr<torch::nn::Module>>(obj.attr("_module"));
        }
        throw;
    }
}

} // anonymous

// ============================================================================
// nn.parallel  ---------------------------------------------------------------
// ============================================================================
void init_parallel_bindings(py::module& m) {
    using torch::nn::parallel::TPConfig;
    using torch::nn::parallel::ColumnParallelLinear;
    using torch::nn::parallel::RowParallelLinear;
    using torch::nn::parallel::Pipeline;

    py::class_<TPConfig>(m, "TPConfig")
        .def(py::init<>())
        .def_readwrite("rank", &TPConfig::rank)
        .def_readwrite("world_size", &TPConfig::world_size)
        .def_readwrite("sync_dir", &TPConfig::sync_dir)
        .def_readwrite("timeout_us", &TPConfig::timeout_us);

    py::class_<ColumnParallelLinear, torch::nn::Module,
               std::shared_ptr<ColumnParallelLinear>>(m, "ColumnParallelLinear")
        .def(py::init<int64_t, int64_t, TPConfig, bool, bool, uint64_t>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("tp_config"), py::arg("gather_output") = true,
             py::arg("bias") = true, py::arg("init_seed") = 1234567ULL)
        .def("forward", &ColumnParallelLinear::forward)
        .def("__call__", &ColumnParallelLinear::forward)
        .def_property_readonly("in_features", &ColumnParallelLinear::in_features)
        .def_property_readonly("out_features", &ColumnParallelLinear::out_features)
        .def_property_readonly("shard_size", &ColumnParallelLinear::shard_size);

    py::class_<RowParallelLinear, torch::nn::Module,
               std::shared_ptr<RowParallelLinear>>(m, "RowParallelLinear")
        .def(py::init<int64_t, int64_t, TPConfig, bool, bool, uint64_t>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("tp_config"), py::arg("input_is_parallel") = true,
             py::arg("bias") = true, py::arg("init_seed") = 7654321ULL)
        .def("forward", &RowParallelLinear::forward)
        .def("__call__", &RowParallelLinear::forward)
        .def_property_readonly("in_features", &RowParallelLinear::in_features)
        .def_property_readonly("out_features", &RowParallelLinear::out_features)
        .def_property_readonly("shard_size", &RowParallelLinear::shard_size);

    // Collectives (free functions)
    m.def("tp_barrier", &torch::nn::parallel::tp_barrier, py::arg("tp_config"));
    m.def("tp_all_gather", &torch::nn::parallel::tp_all_gather,
          py::arg("tensor"), py::arg("gather_dim"), py::arg("tp_config"));
    m.def("tp_all_reduce_sum", &torch::nn::parallel::tp_all_reduce_sum,
          py::arg("tensor"), py::arg("tp_config"));

    // Pipeline
    py::class_<Pipeline, std::shared_ptr<Pipeline>>(m, "Pipeline")
        .def(py::init<std::shared_ptr<torch::nn::Sequential>, int, int>(),
             py::arg("model"), py::arg("num_stages"), py::arg("chunks") = 4)
        .def("forward", &Pipeline::forward)
        .def("__call__", &Pipeline::forward)
        .def_property_readonly("num_stages", &Pipeline::num_stages)
        .def_property_readonly("chunks", &Pipeline::chunks);
}

// ============================================================================
// distributed  ---------------------------------------------------------------
// ============================================================================
void init_distributed_bindings(py::module& m) {
    using torch::distributed::ReduceOp;
    using torch::distributed::BackendType;
    using torch::distributed::ProcessGroup;
    using torch::distributed::ProcessGroupPtr;
    using torch::distributed::DistributedDataParallel;
    using torch::distributed::DDPNoSyncGuard;
    using torch::distributed::FullyShardedDataParallel;
    using torch::distributed::FSDPConfig;
    using torch::distributed::DistArgs;

    py::enum_<ReduceOp>(m, "ReduceOp")
        .value("SUM", ReduceOp::SUM)
        .value("AVG", ReduceOp::AVG)
        .value("MAX", ReduceOp::MAX)
        .value("MIN", ReduceOp::MIN)
        .export_values();

    py::enum_<BackendType>(m, "BackendType")
        .value("SHARED_MEMORY", BackendType::SHARED_MEMORY)
        .value("NCCL",          BackendType::NCCL)
        .export_values();

    py::class_<ProcessGroup, ProcessGroupPtr>(m, "ProcessGroup")
        .def("all_reduce", [](ProcessGroup& self, at::Tensor& t, ReduceOp op) {
            self.all_reduce(t, op);
        }, py::arg("tensor"), py::arg("op") = ReduceOp::SUM)
        .def("broadcast", [](ProcessGroup& self, at::Tensor& t, int src) {
            self.broadcast(t, src);
        }, py::arg("tensor"), py::arg("src") = 0)
        .def("barrier", &ProcessGroup::barrier)
        .def_property_readonly("rank", &ProcessGroup::rank)
        .def_property_readonly("world_size", &ProcessGroup::world_size)
        .def_property_readonly("backend", &ProcessGroup::backend);

    // init_process_group factory: returns the list of per-rank PGs for
    // intra-process multi-rank work (matches SharedMemoryBackend::SharedState).
    m.def("init_process_group",
          &torch::distributed::dist::init_process_group,
          py::arg("backend"), py::arg("world_size"),
          "Create a process group (list of per-rank PGs sharing state).");

    m.def("init_process_group_single_rank",
          &torch::distributed::dist::init_process_group_single_rank,
          py::arg("backend"), py::arg("rank"), py::arg("world_size"),
          "Create a single-rank PG for multi-process launchers.");

    // Legacy helpers
    m.def("init", &torch::distributed::dist::init, py::arg("world_size"));
    m.def("finalize", &torch::distributed::dist::finalize);
    m.def("is_initialized", &torch::distributed::dist::is_initialized);
    m.def("world_size", &torch::distributed::dist::world_size);
    m.def("all_reduce", [](at::Tensor& t, int rank, ReduceOp op) {
              torch::distributed::dist::all_reduce(t, rank, op);
          },
          py::arg("tensor"), py::arg("rank"), py::arg("op") = ReduceOp::SUM);
    m.def("broadcast", &torch::distributed::dist::broadcast,
          py::arg("tensor"), py::arg("rank"), py::arg("src") = 0);
    m.def("scatter", &torch::distributed::dist::scatter,
          py::arg("tensor"), py::arg("rank"));

    // ----- DDP no_sync() context-manager helper -----
    //
    // Python pattern:
    //   with ddp.no_sync():
    //       loss.backward()           # accumulates locally, no AllReduce
    //   loss.backward()               # this one syncs
    //
    // We can't bind the C++ DDPNoSyncGuard directly (it's a stack-RAII type,
    // non-movable, holds a reference). Instead expose a tiny Python-level
    // helper that toggles DistributedDataParallel::set_require_grad_sync()
    // on __enter__/__exit__ — semantically identical to the C++ guard.
    struct PyDDPNoSyncCtx {
        std::shared_ptr<DistributedDataParallel> ddp;
        bool prev = true;
        bool entered = false;
    };
    py::class_<PyDDPNoSyncCtx, std::shared_ptr<PyDDPNoSyncCtx>>(m, "_DDPNoSyncCtx")
        .def("__enter__", [](PyDDPNoSyncCtx& self) {
            if (self.entered) {
                throw std::runtime_error("DDP.no_sync() context already entered");
            }
            self.prev = self.ddp->require_grad_sync();
            self.ddp->set_require_grad_sync(false);
            self.entered = true;
        })
        .def("__exit__", [](PyDDPNoSyncCtx& self,
                            py::object, py::object, py::object) {
            if (self.entered) {
                self.ddp->set_require_grad_sync(self.prev);
                self.entered = false;
            }
            return false;  // don't swallow exceptions
        });

    // DDP
    py::class_<DistributedDataParallel, torch::nn::Module,
               std::shared_ptr<DistributedDataParallel>>(m, "DistributedDataParallel")
        .def(py::init([](py::object module, ProcessGroupPtr pg, bool bcast) {
            return std::make_shared<DistributedDataParallel>(
                to_module_ptr(module), pg, bcast);
        }), py::arg("module"), py::arg("process_group"),
            py::arg("broadcast_parameters") = true)
        .def("forward",
             py::overload_cast<const at::Tensor&>(&DistributedDataParallel::forward))
        .def("__call__",
             py::overload_cast<const at::Tensor&>(&DistributedDataParallel::forward))
        .def("finish_gradient_synchronization",
             &DistributedDataParallel::finish_gradient_synchronization)
        .def("sync_gradients", &DistributedDataParallel::sync_gradients)
        // ---- no_sync() / require_grad_sync — gradient accumulation ----
        // Python:
        //   with ddp.no_sync():
        //       for mb in micro_batches[:-1]:
        //           loss_fn(ddp(mb.x), mb.y).backward()   # local accum only
        //   # final micro-batch outside the guard:
        //   loss_fn(ddp(last.x), last.y).backward()
        //   ddp.finish_gradient_synchronization()         # one AllReduce
        //   optim.step()
        // Saves N-1 AllReduces per N-step accumulation.
        .def_property("require_grad_sync",
                      &DistributedDataParallel::require_grad_sync,
                      &DistributedDataParallel::set_require_grad_sync)
        .def("no_sync",
             [](std::shared_ptr<DistributedDataParallel> self) {
                 auto ctx = std::make_shared<PyDDPNoSyncCtx>();
                 ctx->ddp = std::move(self);
                 return ctx;
             },
             "Context manager that suppresses gradient AllReduce within "
             "the with-block. Use across the first N-1 of N gradient "
             "accumulation micro-batches; the Nth backward (without "
             "no_sync) plus a single finish_gradient_synchronization() "
             "averages the accumulated grad across all ranks.")
        .def_property_readonly("module", &DistributedDataParallel::module)
        .def_property_readonly("process_group",
            &DistributedDataParallel::process_group);

    // FSDP config + wrapper
    py::class_<FSDPConfig> fsdp_cfg(m, "FSDPConfig");
    py::enum_<FSDPConfig::ShardingStrategy>(fsdp_cfg, "ShardingStrategy")
        .value("FULL_SHARD",     FSDPConfig::ShardingStrategy::FULL_SHARD)
        .value("SHARD_GRAD_OP",  FSDPConfig::ShardingStrategy::SHARD_GRAD_OP)
        .value("NO_SHARD",       FSDPConfig::ShardingStrategy::NO_SHARD)
        .export_values();

    fsdp_cfg.def(py::init<>())
        .def_readwrite("rank", &FSDPConfig::rank)
        .def_readwrite("world_size", &FSDPConfig::world_size)
        .def_readwrite("sync_dir", &FSDPConfig::sync_dir)
        .def_readwrite("timeout_ms", &FSDPConfig::timeout_ms)
        .def_readwrite("poll_us", &FSDPConfig::poll_us)
        .def_readwrite("strategy", &FSDPConfig::strategy);

    py::class_<FullyShardedDataParallel, torch::nn::Module,
               std::shared_ptr<FullyShardedDataParallel>>(m, "FullyShardedDataParallel")
        .def(py::init([](py::object module, const FSDPConfig& cfg) {
            return std::make_shared<FullyShardedDataParallel>(
                to_module_ptr(module), cfg);
        }), py::arg("module"), py::arg("config"))
        .def("forward",
             py::overload_cast<const at::Tensor&>(&FullyShardedDataParallel::forward))
        .def("__call__",
             py::overload_cast<const at::Tensor&>(&FullyShardedDataParallel::forward))
        .def("all_gather_params", &FullyShardedDataParallel::all_gather_params)
        .def("reshard_params", &FullyShardedDataParallel::reshard_params)
        .def("reduce_scatter_grads",
             &FullyShardedDataParallel::reduce_scatter_grads)
        .def_property_readonly("rank", &FullyShardedDataParallel::rank)
        .def_property_readonly("world_size",
             &FullyShardedDataParallel::world_size);

    // Launcher
    py::class_<DistArgs>(m, "DistArgs")
        .def(py::init<>())
        .def_readwrite("rank", &DistArgs::rank)
        .def_readwrite("world_size", &DistArgs::world_size)
        .def_readwrite("master_addr", &DistArgs::master_addr)
        .def_readwrite("master_port", &DistArgs::master_port);

    m.def("launch", [](int world_size,
                       py::function worker_fn,
                       const std::string& master_addr,
                       int master_port) {
        // Note: on non-POSIX platforms this returns -1.
        return torch::distributed::launch(
            world_size,
            [worker_fn](int rank, int ws) -> int {
                py::gil_scoped_acquire gil;
                try {
                    py::object rv = worker_fn(rank, ws);
                    return rv.is_none() ? 0 : rv.cast<int>();
                } catch (const std::exception& e) {
                    return 1;
                }
            },
            master_addr, master_port);
    }, py::arg("world_size"), py::arg("worker_fn"),
       py::arg("master_addr") = std::string("127.0.0.1"),
       py::arg("master_port") = 29500);
}

// ============================================================================
// trainer  -------------------------------------------------------------------
// ============================================================================
namespace {

// Trampoline so Python subclasses can override training_step / validation_step
// / configure_optimizer.
class PyLightningModule : public torch::trainer::LightningModule {
public:
    using torch::trainer::LightningModule::LightningModule;

    at::Tensor training_step(const at::Tensor& batch, int batch_idx) override {
        PYBIND11_OVERRIDE_PURE(at::Tensor, torch::trainer::LightningModule,
                               training_step, batch, batch_idx);
    }
    at::Tensor validation_step(const at::Tensor& batch, int batch_idx) override {
        PYBIND11_OVERRIDE(at::Tensor, torch::trainer::LightningModule,
                          validation_step, batch, batch_idx);
    }
    std::shared_ptr<torch::optim::Optimizer> configure_optimizer() override {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<torch::optim::Optimizer>,
                               torch::trainer::LightningModule,
                               configure_optimizer, );
    }
};

} // anon

void init_trainer_bindings(py::module& m) {
    using torch::trainer::TrainerConfig;
    using torch::trainer::Trainer;
    using torch::trainer::LightningModule;

    py::class_<TrainerConfig>(m, "TrainerConfig")
        .def(py::init<>())
        .def_readwrite("max_epochs", &TrainerConfig::max_epochs)
        .def_readwrite("log_every_n_steps", &TrainerConfig::log_every_n_steps)
        .def_readwrite("val_check_interval", &TrainerConfig::val_check_interval)
        .def_readwrite("checkpoint_dir", &TrainerConfig::checkpoint_dir)
        .def_readwrite("save_every_n_epochs", &TrainerConfig::save_every_n_epochs)
        .def_readwrite("enable_progress_bar", &TrainerConfig::enable_progress_bar)
        .def_readwrite("gradient_clip_val", &TrainerConfig::gradient_clip_val)
        .def_readwrite("accumulate_grad_batches",
                       &TrainerConfig::accumulate_grad_batches);

    py::class_<LightningModule, torch::nn::Module, PyLightningModule,
               std::shared_ptr<LightningModule>>(m, "LightningModule")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("name"))
        .def("training_step", &LightningModule::training_step,
             py::arg("batch"), py::arg("batch_idx"))
        .def("validation_step", &LightningModule::validation_step,
             py::arg("batch"), py::arg("batch_idx"))
        .def("configure_optimizer", &LightningModule::configure_optimizer);

    py::class_<Trainer>(m, "Trainer")
        .def(py::init<TrainerConfig>(), py::arg("config"))
        .def("save_checkpoint", &Trainer::save_checkpoint,
             py::arg("module"), py::arg("path"))
        .def("load_checkpoint", &Trainer::load_checkpoint,
             py::arg("module"), py::arg("path"))
        .def_property_readonly("global_step", &Trainer::global_step)
        .def_property_readonly("config", &Trainer::config);

    // We don't bind fit()/test() directly because DataLoader is templated on
    // Dataset type and the Python-side wrapper implements the loop by calling
    // training_step / validation_step repeatedly through the Python iterator
    // protocol — this is more flexible than binding a fixed DataLoader type.
}

// ============================================================================
// onnx / mlir / mobile / jit  ------------------------------------------------
// ============================================================================
void init_onnx_bindings(py::module& m) {
    m.def("export", [](py::object model, const at::Tensor& example_input,
                       const std::string& path,
                       const std::string& input_name,
                       const std::string& output_name) {
        auto mp = to_module_ptr(model);
        return torch::onnx::export_model(*mp, example_input, path,
                                          input_name, output_name);
    }, py::arg("model"), py::arg("example_input"), py::arg("path"),
       py::arg("input_name") = std::string("input"),
       py::arg("output_name") = std::string("output"),
       "Export a promethorch.nn.Module to an ONNX file. Returns True on success.");

    m.def("self_test", &torch::onnx::onnx_self_test,
          py::arg("tmp_path") = std::string("/tmp/test.onnx"));
}

void init_mlir_bindings(py::module& m) {
    m.def("export", [](py::object model,
                       const std::vector<int64_t>& input_shape,
                       const std::string& path) {
        auto mp = to_module_ptr(model);
        auto seq = std::dynamic_pointer_cast<torch::nn::Sequential>(mp);
        if (!seq) {
            throw std::runtime_error(
                "mlir.export: model must be a Sequential container");
        }
        std::string text = torch::mlir::export_mlir(*seq, input_shape, path);
        return !text.empty();
    }, py::arg("model"), py::arg("input_shape"), py::arg("path"),
       "Export a Sequential to MLIR text form. Returns True on success.");
}

void init_mobile_bindings(py::module& m) {
    m.def("export", [](py::object model, const at::Tensor& example_input,
                       const std::string& path) {
        auto mp = to_module_ptr(model);
        // Mobile exporter accepts Sequential specifically — cast or error.
        auto seq = std::dynamic_pointer_cast<torch::nn::Sequential>(mp);
        if (!seq) {
            throw std::runtime_error(
                "mobile.export: model must be a Sequential container");
        }
        return torch::mobile::export_model(*seq, example_input, path);
    }, py::arg("model"), py::arg("example_input"), py::arg("path"));

    py::class_<torch::mobile::MobileExecutor>(m, "MobileExecutor")
        .def(py::init<>())
        .def("load", [](torch::mobile::MobileExecutor& self, const std::string& path) {
            return self.load(path);
        }, py::arg("path"))
        .def("forward", &torch::mobile::MobileExecutor::forward,
             py::arg("input"))
        .def("__call__", &torch::mobile::MobileExecutor::forward);
}

void init_jit_bindings(py::module& m) {
    // The IR-less replay prototype has a Python-side friendlier wrapper —
    // expose just the traced-ops facade here via a simple no-op compile()
    // that returns the callable unchanged (matches the Python fallback).
    m.def("compile", [](py::function fn, py::object /*example_input*/) {
        // Real tracing requires the user to call jit::traced_* variants from
        // inside fn, which is impossible to do from generic Python code.
        // We therefore return the function unchanged; this matches PyTorch's
        // semantics for "torch.compile on unsupported callables".
        return fn;
    }, py::arg("fn"), py::arg("example_input") = py::none(),
       "Stub: returns fn unchanged (tracing requires jit::traced_* builder API).");
}

// ============================================================================
// vision  --------------------------------------------------------------------
// ============================================================================
void init_vision_bindings(py::module& m) {
    using torch::vision::ImageFolder;
    using torch::vision::transforms::Transform;
    using torch::vision::transforms::TransformPtr;
    using torch::vision::transforms::Compose;
    using torch::vision::transforms::ToTensor;

    // Transforms
    py::module tm = m.def_submodule("transforms", "Image transforms");

    py::class_<Transform, TransformPtr>(tm, "Transform")
        .def("__call__", &Transform::operator());

    py::class_<Compose, Transform, std::shared_ptr<Compose>>(tm, "Compose")
        .def(py::init<>())
        .def(py::init<std::vector<TransformPtr>>(), py::arg("transforms"))
        .def("push_back", &Compose::push_back)
        .def("__call__", &Compose::operator());

    py::class_<ToTensor, Transform, std::shared_ptr<ToTensor>>(tm, "ToTensor")
        .def(py::init<>())
        .def("__call__", &ToTensor::operator());

    // ImageFolder
    py::class_<ImageFolder, std::shared_ptr<ImageFolder>>(m, "ImageFolder")
        .def(py::init([](const std::string& root, TransformPtr transform) {
            ImageFolder::TransformFn fn = nullptr;
            if (transform) {
                fn = [transform](const at::Tensor& x) { return (*transform)(x); };
            }
            return std::make_shared<ImageFolder>(root, fn);
        }), py::arg("root"), py::arg("transform") = TransformPtr())
        .def("__len__", &ImageFolder::size)
        .def("__getitem__", [](ImageFolder& self, size_t i) {
            auto ex = self.get(i);
            return py::make_tuple(ex.data, ex.target);
        })
        .def_property_readonly("classes", &ImageFolder::classes);

    // MobileNetV2 — helper factory (returns shared Module ptr)
    m.def("mobilenet_v2", [](int64_t num_classes, double width_mult) {
        auto model = std::make_shared<torch::vision::models::MobileNetV2>(
            num_classes, width_mult);
        return std::dynamic_pointer_cast<torch::nn::Module>(model);
    }, py::arg("num_classes") = 1000, py::arg("width_mult") = 1.0);
}

// ============================================================================
// quantization  --------------------------------------------------------------
// ============================================================================
void init_quantization_bindings(py::module& m) {
    using torch::quantization::QuantizedLinear;

    py::class_<QuantizedLinear, torch::nn::Module,
               std::shared_ptr<QuantizedLinear>>(m, "QuantizedLinear")
        .def(py::init<int64_t, int64_t, bool>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("bias") = true)
        .def("forward", &QuantizedLinear::forward)
        .def("__call__", &QuantizedLinear::forward);

    m.def("fake_quantize", &torch::quantization::fake_quantize_qdq,
          py::arg("input"), py::arg("scale"), py::arg("zero_point"),
          py::arg("qmin") = -128, py::arg("qmax") = 127);

    m.def("prepare_qat", [](py::object model) {
        // In-place replacement of nn::Linear with QuantizedLinear.
        auto mp = to_module_ptr(model);
        torch::quantization::prepare_qat(*mp);
        return model;
    }, py::arg("model"),
       "Replace Linear submodules with QuantizedLinear (returns same object).");

    m.def("convert", [](py::object model) {
        auto mp = to_module_ptr(model);
        torch::quantization::convert(*mp);
        return model;
    }, py::arg("model"),
       "Freeze QAT observers (no further scale updates).");
}

// ============================================================================
// autograd (forward-mode + vmap)  --------------------------------------------
// ============================================================================
void init_autograd_extra_bindings(py::module& m) {
    using torch::autograd::forward_ad::DualLevel;

    py::class_<DualLevel>(m, "DualLevel")
        .def(py::init<>())
        .def("__enter__", [](DualLevel& self) -> DualLevel& { return self; })
        .def("__exit__", [](DualLevel&, py::args) {});

    m.def("make_dual", &torch::autograd::forward_ad::make_dual,
          py::arg("primal"), py::arg("tangent"));
    m.def("unpack_dual", [](const at::Tensor& t) {
        auto u = torch::autograd::forward_ad::unpack_dual(t);
        return py::make_tuple(u.primal, u.tangent);
    }, py::arg("tensor"));

    m.def("jvp", [](py::function f, const at::Tensor& primal,
                    const at::Tensor& tangent) {
        // Call the user fn under a DualLevel with (primal, tangent) attached.
        DualLevel level;
        at::Tensor dual = torch::autograd::forward_ad::make_dual(primal, tangent);
        py::object result = f(dual);
        at::Tensor out = result.cast<at::Tensor>();
        auto u = torch::autograd::forward_ad::unpack_dual(out);
        return py::make_tuple(u.primal, u.tangent);
    }, py::arg("f"), py::arg("primal"), py::arg("tangent"),
       "Compute (f(primal), Jf @ tangent) using forward-mode AD.");

    m.def("vmap", [](py::function f, const at::Tensor& input,
                     int64_t in_dim, int64_t out_dim) {
        return torch::autograd::vmap(
            [f](at::Tensor x) -> at::Tensor {
                py::gil_scoped_acquire gil;
                py::object out = f(x);
                return out.cast<at::Tensor>();
            },
            input, in_dim, out_dim);
    }, py::arg("f"), py::arg("input"),
       py::arg("in_dim") = 0, py::arg("out_dim") = 0,
       "Vectorized map: apply f over slices of `input` along `in_dim`.");
}

// ============================================================================
// serve (LLM engine scaffold) ------------------------------------------------
// ============================================================================
// We don't have a dedicated torch::serve C++ namespace yet, so this submodule
// just exposes a thin ModelRunner that loads a state dict and runs a
// user-supplied forward. The full LLMEngine (prompt caching, batching,
// tokenization) lives Python-side in promethorch/serve/engine.py and uses
// this as its low-level executor.
void init_serve_bindings(py::module& m) {
    m.def("_load_state_dict", [](const std::string& path) {
        return torch::load_state_dict(path);
    }, py::arg("path"));

    m.def("_save_state_dict", [](const std::unordered_map<std::string, at::Tensor>& sd,
                                 const std::string& path) {
        torch::save_state_dict(sd, path);
    }, py::arg("state_dict"), py::arg("path"));
}

// ============================================================================
// Entry point called from init.cpp
// ============================================================================
void init_new_bindings(py::module& m) {
    auto parallel_m     = m.def_submodule("parallel",
        "Tensor + pipeline parallelism");
    auto distributed_m  = m.def_submodule("distributed",
        "Distributed training: DDP, FSDP, process groups, launcher");
    auto trainer_m      = m.def_submodule("trainer",
        "Lightning-style Trainer");
    auto onnx_m         = m.def_submodule("onnx_export", "ONNX export");
    auto mlir_m         = m.def_submodule("mlir_export", "MLIR export");
    auto mobile_m       = m.def_submodule("mobile_export", "ExecuTorch-like export");
    auto jit_m          = m.def_submodule("jit_compile",  "JIT compile");
    auto vision_m       = m.def_submodule("vision",       "Vision datasets / models");
    auto quant_m        = m.def_submodule("quantization", "QAT / INT8");
    auto autograd_m     = m.def_submodule("autograd_fwd",
        "Forward-mode AD + vmap");
    auto serve_m        = m.def_submodule("serve",        "LLM inference engine");

    init_parallel_bindings(parallel_m);
    init_distributed_bindings(distributed_m);
    init_trainer_bindings(trainer_m);
    init_onnx_bindings(onnx_m);
    init_mlir_bindings(mlir_m);
    init_mobile_bindings(mobile_m);
    init_jit_bindings(jit_m);
    init_vision_bindings(vision_m);
    init_quantization_bindings(quant_m);
    init_autograd_extra_bindings(autograd_m);
    init_serve_bindings(serve_m);
}
