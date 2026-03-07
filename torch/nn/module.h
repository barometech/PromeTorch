#pragma once

#include "torch/nn/parameter.h"
#include "aten/src/ATen/ATen.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <functional>
#include <sstream>
#include <iomanip>
#include <atomic>
#include <map>

namespace torch {
namespace nn {

using at::Tensor;

// Forward declarations
class Module;
using ModulePtr = std::shared_ptr<Module>;

// ============================================================================
// Hook Types
// ============================================================================

using ForwardPreHook = std::function<void(Module&, const Tensor&)>;
using ForwardHook = std::function<void(Module&, const Tensor&, const Tensor&)>;
using ForwardHookWithReturn = std::function<Tensor(Module&, const Tensor&, const Tensor&)>;

// Handle for removing hooks
struct HookHandle {
    int64_t id;
    explicit HookHandle(int64_t id_) : id(id_) {}
};

// ============================================================================
// Module - Base Class for All Neural Network Modules
// ============================================================================
// This is the base class for all neural network modules in PromeTorch.
// Your models should subclass this class.
//
// Modules can contain:
// - Parameters (weights that are trained)
// - Buffers (state that is not trained, like running mean in BatchNorm)
// - Submodules (nested modules)
//
// Key methods to override:
// - forward(): Define the computation
// - reset_parameters(): Initialize parameters (called automatically)

class Module : public std::enable_shared_from_this<Module> {
public:
    Module() : training_(true), name_("Module") {}
    explicit Module(const std::string& name) : training_(true), name_(name) {}
    virtual ~Module() = default;

    // ========================================================================
    // Forward Pass
    // ========================================================================

    // Override this method to define the forward computation
    virtual Tensor forward(const Tensor& input) {
        throw std::runtime_error("forward() not implemented for " + name_);
    }

    // Multi-input forward (override for modules that need multiple inputs)
    virtual Tensor forward(const Tensor& input1, const Tensor& input2) {
        throw std::runtime_error("forward(input1, input2) not implemented for " + name_);
    }

    // Variadic forward
    virtual Tensor forward(const std::vector<Tensor>& inputs) {
        if (inputs.size() == 1) {
            return forward(inputs[0]);
        } else if (inputs.size() == 2) {
            return forward(inputs[0], inputs[1]);
        }
        throw std::runtime_error("forward(vector) not implemented for " + name_);
    }

    // Operator() calls forward with hooks
    Tensor operator()(const Tensor& input) {
        // Pre-hooks
        for (auto& [id, hook] : forward_pre_hooks_) {
            hook(*this, input);
        }
        // Forward
        Tensor output = forward(input);
        // Post-hooks
        for (auto& [id, hook] : forward_hooks_) {
            hook(*this, input, output);
        }
        // Post-hooks with return (can modify output)
        for (auto& [id, hook] : forward_hooks_with_return_) {
            Tensor new_output = hook(*this, input, output);
            if (new_output.defined()) {
                output = new_output;
            }
        }
        return output;
    }

    Tensor operator()(const Tensor& input1, const Tensor& input2) {
        return forward(input1, input2);
    }

    // ========================================================================
    // Parameter Management
    // ========================================================================

    // Register a parameter
    void register_parameter(const std::string& name, Parameter param) {
        parameters_[name] = std::move(param);
        parameter_order_.push_back(name);
    }

    // Register a buffer
    void register_buffer(const std::string& name, Buffer buffer) {
        buffers_[name] = std::move(buffer);
        buffer_order_.push_back(name);
    }

    // Register a submodule
    void register_module(const std::string& name, ModulePtr module) {
        if (module) {
            module->name_ = name;
        }
        submodules_[name] = std::move(module);
        submodule_order_.push_back(name);
    }

    // Get parameter by name
    Parameter* get_parameter(const std::string& name) {
        auto it = parameters_.find(name);
        return it != parameters_.end() ? &it->second : nullptr;
    }

    const Parameter* get_parameter(const std::string& name) const {
        auto it = parameters_.find(name);
        return it != parameters_.end() ? &it->second : nullptr;
    }

    // Get buffer by name
    Buffer* get_buffer(const std::string& name) {
        auto it = buffers_.find(name);
        return it != buffers_.end() ? &it->second : nullptr;
    }

    // Get submodule by name
    ModulePtr get_submodule(const std::string& name) {
        auto it = submodules_.find(name);
        return it != submodules_.end() ? it->second : nullptr;
    }

    // ========================================================================
    // Iterators for Parameters/Modules
    // ========================================================================

    // Get all parameters (including submodules)
    std::vector<Parameter*> parameters(bool recurse = true) {
        std::vector<Parameter*> result;
        for (const auto& name : parameter_order_) {
            result.push_back(&parameters_[name]);
        }
        if (recurse) {
            for (const auto& name : submodule_order_) {
                auto& submodule = submodules_[name];
                if (submodule) {
                    auto sub_params = submodule->parameters(true);
                    result.insert(result.end(), sub_params.begin(), sub_params.end());
                }
            }
        }
        return result;
    }

    // Get all named parameters
    std::vector<std::pair<std::string, Parameter*>> named_parameters(
        const std::string& prefix = "",
        bool recurse = true
    ) {
        std::vector<std::pair<std::string, Parameter*>> result;
        std::string actual_prefix = prefix.empty() ? "" : prefix + ".";

        for (const auto& name : parameter_order_) {
            result.emplace_back(actual_prefix + name, &parameters_[name]);
        }

        if (recurse) {
            for (const auto& name : submodule_order_) {
                auto& submodule = submodules_[name];
                if (submodule) {
                    auto sub_params = submodule->named_parameters(actual_prefix + name, true);
                    result.insert(result.end(), sub_params.begin(), sub_params.end());
                }
            }
        }
        return result;
    }

    // Get all buffers
    std::vector<Buffer*> buffers(bool recurse = true) {
        std::vector<Buffer*> result;
        for (const auto& name : buffer_order_) {
            result.push_back(&buffers_[name]);
        }
        if (recurse) {
            for (const auto& name : submodule_order_) {
                auto& submodule = submodules_[name];
                if (submodule) {
                    auto sub_buffers = submodule->buffers(true);
                    result.insert(result.end(), sub_buffers.begin(), sub_buffers.end());
                }
            }
        }
        return result;
    }

    // Get all submodules
    std::vector<ModulePtr> modules(bool recurse = true) {
        std::vector<ModulePtr> result;
        result.push_back(shared_from_this());

        for (const auto& name : submodule_order_) {
            auto& submodule = submodules_[name];
            if (submodule) {
                if (recurse) {
                    auto sub_modules = submodule->modules(true);
                    result.insert(result.end(), sub_modules.begin(), sub_modules.end());
                } else {
                    result.push_back(submodule);
                }
            }
        }
        return result;
    }

    // Get direct children
    std::vector<ModulePtr> children() {
        std::vector<ModulePtr> result;
        for (const auto& name : submodule_order_) {
            if (submodules_[name]) {
                result.push_back(submodules_[name]);
            }
        }
        return result;
    }

    // ========================================================================
    // Training Mode
    // ========================================================================

    // Set training mode
    void train(bool mode = true) {
        training_ = mode;
        for (auto& [name, submodule] : submodules_) {
            if (submodule) {
                submodule->train(mode);
            }
        }
    }

    // Set evaluation mode
    void eval() {
        train(false);
    }

    // Check training mode
    bool is_training() const {
        return training_;
    }

    // ========================================================================
    // Gradient Operations
    // ========================================================================

    // Zero all gradients
    void zero_grad(bool set_to_none = false) {
        for (auto& [name, param] : parameters_) {
            if (set_to_none) {
                auto* meta = autograd::get_autograd_meta(param.data());
                if (meta) {
                    meta->grad_ = nullptr;
                }
            } else {
                param.zero_grad();
            }
        }
        for (auto& [name, submodule] : submodules_) {
            if (submodule) {
                submodule->zero_grad(set_to_none);
            }
        }
    }

    // Require gradients for all parameters
    void requires_grad_(bool requires_grad = true) {
        for (auto& [name, param] : parameters_) {
            param.set_requires_grad(requires_grad);
        }
        for (auto& [name, submodule] : submodules_) {
            if (submodule) {
                submodule->requires_grad_(requires_grad);
            }
        }
    }

    // ========================================================================
    // Device/Dtype Operations
    // ========================================================================

    // Move to device
    void to(c10::Device device) {
        for (auto& [name, param] : parameters_) {
            if (param.defined()) {
                param.set_data(param.data().to(device));
            }
        }
        for (auto& [name, buffer] : buffers_) {
            if (buffer.defined()) {
                buffer.set_data(buffer.data().to(device));
            }
        }
        for (auto& [name, submodule] : submodules_) {
            if (submodule) {
                submodule->to(device);
            }
        }
    }

    // Convert to dtype
    void to(c10::ScalarType dtype) {
        for (auto& [name, param] : parameters_) {
            if (param.defined()) {
                param.set_data(param.data().to(dtype));
            }
        }
        for (auto& [name, buffer] : buffers_) {
            if (buffer.defined()) {
                buffer.set_data(buffer.data().to(dtype));
            }
        }
        for (auto& [name, submodule] : submodules_) {
            if (submodule) {
                submodule->to(dtype);
            }
        }
    }

    // Move to CPU
    void cpu() {
        to(c10::Device(c10::DeviceType::CPU));
    }

    // Move to CUDA
    void cuda(int device_index = 0) {
        to(c10::Device(c10::DeviceType::CUDA, static_cast<int8_t>(device_index)));
    }

    // Convert to float
    void float_() {
        to(c10::ScalarType::Float);
    }

    // Convert to double
    void double_() {
        to(c10::ScalarType::Double);
    }

    // Convert to half
    void half() {
        to(c10::ScalarType::Half);
    }

    // ========================================================================
    // State Dict
    // ========================================================================

    using StateDict = std::unordered_map<std::string, Tensor>;

    // Get state dictionary
    StateDict state_dict(const std::string& prefix = "") const {
        StateDict state;
        std::string actual_prefix = prefix.empty() ? "" : prefix + ".";

        // Add parameters
        for (const auto& name : parameter_order_) {
            const auto& param = parameters_.at(name);
            if (param.defined()) {
                state[actual_prefix + name] = param.data();
            }
        }

        // Add persistent buffers
        for (const auto& name : buffer_order_) {
            const auto& buffer = buffers_.at(name);
            if (buffer.defined() && buffer.persistent()) {
                state[actual_prefix + name] = buffer.data();
            }
        }

        // Add submodule states
        for (const auto& name : submodule_order_) {
            const auto& submodule = submodules_.at(name);
            if (submodule) {
                auto sub_state = submodule->state_dict(actual_prefix + name);
                state.insert(sub_state.begin(), sub_state.end());
            }
        }

        return state;
    }

    // Load state dictionary
    void load_state_dict(const StateDict& state_dict, bool strict = true) {
        std::vector<std::string> missing_keys;
        std::vector<std::string> unexpected_keys;

        // Load parameters
        for (auto& [name, param] : parameters_) {
            auto it = state_dict.find(name);
            if (it != state_dict.end()) {
                param.set_data(it->second.clone());
            } else if (strict) {
                missing_keys.push_back(name);
            }
        }

        // Load buffers
        for (auto& [name, buffer] : buffers_) {
            if (!buffer.persistent()) continue;
            auto it = state_dict.find(name);
            if (it != state_dict.end()) {
                buffer.set_data(it->second.clone());
            } else if (strict) {
                missing_keys.push_back(name);
            }
        }

        // Load submodules
        for (auto& [name, submodule] : submodules_) {
            if (submodule) {
                StateDict sub_state;
                std::string sub_prefix = name + ".";
                for (const auto& [key, value] : state_dict) {
                    if (key.rfind(sub_prefix, 0) == 0) {
                        sub_state[key.substr(sub_prefix.size())] = value;
                    }
                }
                submodule->load_state_dict(sub_state, strict);
            }
        }

        if (strict && !missing_keys.empty()) {
            std::string msg = "Missing keys: ";
            for (const auto& key : missing_keys) {
                msg += key + ", ";
            }
            throw std::runtime_error(msg);
        }
    }

    // ========================================================================
    // Parameter Count
    // ========================================================================

    int64_t num_parameters(bool only_trainable = false) const {
        int64_t count = 0;
        for (const auto& [name, param] : parameters_) {
            if (!only_trainable || param.requires_grad()) {
                count += param.numel();
            }
        }
        for (const auto& [name, submodule] : submodules_) {
            if (submodule) {
                count += submodule->num_parameters(only_trainable);
            }
        }
        return count;
    }

    // ========================================================================
    // String Representation
    // ========================================================================

    virtual std::string extra_repr() const {
        return "";
    }

    std::string repr(int indent = 0) const {
        std::ostringstream ss;
        std::string indent_str(indent, ' ');

        ss << name_;
        std::string extra = extra_repr();
        if (!extra.empty()) {
            ss << "(" << extra << ")";
        }

        if (!submodule_order_.empty()) {
            ss << "(\n";
            for (const auto& name : submodule_order_) {
                const auto& submodule = submodules_.at(name);
                if (submodule) {
                    ss << indent_str << "  (" << name << "): ";
                    ss << submodule->repr(indent + 2) << "\n";
                }
            }
            ss << indent_str << ")";
        }

        return ss.str();
    }

    // Name
    virtual std::string name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }

    // ========================================================================
    // Hooks
    // ========================================================================

    HookHandle register_forward_pre_hook(ForwardPreHook hook) {
        int64_t id = next_hook_id_++;
        forward_pre_hooks_[id] = std::move(hook);
        return HookHandle(id);
    }

    HookHandle register_forward_hook(ForwardHook hook) {
        int64_t id = next_hook_id_++;
        forward_hooks_[id] = std::move(hook);
        return HookHandle(id);
    }

    HookHandle register_forward_hook_with_return(ForwardHookWithReturn hook) {
        int64_t id = next_hook_id_++;
        forward_hooks_with_return_[id] = std::move(hook);
        return HookHandle(id);
    }

    void remove_hook(const HookHandle& handle) {
        forward_pre_hooks_.erase(handle.id);
        forward_hooks_.erase(handle.id);
        forward_hooks_with_return_.erase(handle.id);
    }

    // ========================================================================
    // Apply Function to All Modules
    // ========================================================================

    void apply(std::function<void(Module&)> fn) {
        fn(*this);
        for (auto& [name, submodule] : submodules_) {
            if (submodule) {
                submodule->apply(fn);
            }
        }
    }

protected:
    // Override to initialize parameters
    virtual void reset_parameters() {}

    bool training_;
    std::string name_;

    // Parameters (ordered for deterministic iteration)
    std::unordered_map<std::string, Parameter> parameters_;
    std::vector<std::string> parameter_order_;

    // Buffers
    std::unordered_map<std::string, Buffer> buffers_;
    std::vector<std::string> buffer_order_;

    // Submodules
    std::unordered_map<std::string, ModulePtr> submodules_;
    std::vector<std::string> submodule_order_;

    // Hooks
    std::map<int64_t, ForwardPreHook> forward_pre_hooks_;
    std::map<int64_t, ForwardHook> forward_hooks_;
    std::map<int64_t, ForwardHookWithReturn> forward_hooks_with_return_;
    inline static std::atomic<int64_t> next_hook_id_{0};
};

// ============================================================================
// Helper function to create modules
// ============================================================================

template<typename T, typename... Args>
std::shared_ptr<T> make_module(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

} // namespace nn
} // namespace torch
