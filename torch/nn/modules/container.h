#pragma once

#include "torch/nn/module.h"
#include <initializer_list>
#include <utility>

namespace torch {
namespace nn {

// ============================================================================
// Sequential - A sequential container
// ============================================================================
// Modules will be added to it in the order they are passed to the constructor.
// The forward() method of Sequential accepts any input and forwards it to
// the first module. It then chains outputs to inputs sequentially for each
// subsequent module, finally returning the output of the last module.

class Sequential : public Module {
public:
    Sequential() : Module("Sequential") {}

    // Construct with initializer list of modules
    Sequential(std::initializer_list<std::pair<std::string, ModulePtr>> modules)
        : Module("Sequential") {
        for (const auto& [name, module] : modules) {
            push_back(name, module);
        }
    }

    // Construct with vector of modules (auto-numbered)
    explicit Sequential(std::initializer_list<ModulePtr> modules)
        : Module("Sequential") {
        int idx = 0;
        for (const auto& module : modules) {
            push_back(std::to_string(idx++), module);
        }
    }

    // Add module with name
    void push_back(const std::string& name, ModulePtr module) {
        register_module(name, std::move(module));
        cache_dirty_ = true;
    }

    // Add module with auto-generated name
    void push_back(ModulePtr module) {
        push_back(std::to_string(submodule_order_.size()), std::move(module));
    }

    // Alias for push_back
    void add(ModulePtr module) {
        push_back(std::move(module));
    }

    // Add module using << operator
    Sequential& operator<<(ModulePtr module) {
        push_back(std::move(module));
        return *this;
    }

    // Get number of modules
    size_t size() const {
        return submodule_order_.size();
    }

    // Check if empty
    bool empty() const {
        return submodule_order_.empty();
    }

    // Access by index
    ModulePtr operator[](size_t index) {
        if (index >= submodule_order_.size()) {
            throw std::out_of_range("Sequential index out of range");
        }
        return submodules_[submodule_order_[index]];
    }

    // Forward pass - chain modules
    // Uses a cached module pointer vector for O(1) iteration (no map lookups).
    // Cache is rebuilt lazily when modules are added or replaced.
    Tensor forward(const Tensor& input) override {
        if (cache_dirty_) rebuild_cache();
        Tensor output = input;
        for (auto* mod : cached_ptrs_) {
            output = mod->forward(output);
        }
        return output;
    }

    std::string extra_repr() const override {
        return "";
    }

private:
    // Rebuild the raw pointer cache from the canonical submodule_order_ + submodules_
    void rebuild_cache() {
        cached_ptrs_.clear();
        cached_ptrs_.reserve(submodule_order_.size());
        for (const auto& name : submodule_order_) {
            cached_ptrs_.push_back(submodules_[name].get());
        }
        cache_dirty_ = false;
    }

    // Cached raw pointers for zero-overhead forward iteration.
    // Lifetime managed by submodules_ (shared_ptr keeps modules alive).
    std::vector<Module*> cached_ptrs_;
    bool cache_dirty_ = true;
};

// ============================================================================
// ModuleList - Holds submodules in a list
// ============================================================================
// ModuleList can be indexed like a regular Python list, but modules it
// contains are properly registered, and will be visible by all Module methods.

class ModuleList : public Module {
public:
    ModuleList() : Module("ModuleList") {}

    explicit ModuleList(std::initializer_list<ModulePtr> modules)
        : Module("ModuleList") {
        for (const auto& module : modules) {
            append(module);
        }
    }

    // Add a module to the end
    void append(ModulePtr module) {
        std::string name = std::to_string(size());
        register_module(name, std::move(module));
    }

    // Extend with multiple modules
    void extend(const std::vector<ModulePtr>& modules) {
        for (const auto& module : modules) {
            append(module);
        }
    }

    // Insert at position
    void insert(size_t index, ModulePtr module) {
        // This is simplified - in reality we'd need to renumber
        if (index > size()) {
            throw std::out_of_range("ModuleList insert index out of range");
        }
        std::string name = std::to_string(index);
        register_module(name, std::move(module));
    }

    // Get size
    size_t size() const {
        return submodule_order_.size();
    }

    // Check if empty
    bool empty() const {
        return submodule_order_.empty();
    }

    // Access by index
    ModulePtr operator[](size_t index) {
        if (index >= submodule_order_.size()) {
            throw std::out_of_range("ModuleList index out of range");
        }
        return submodules_[submodule_order_[index]];
    }

    const ModulePtr operator[](size_t index) const {
        if (index >= submodule_order_.size()) {
            throw std::out_of_range("ModuleList index out of range");
        }
        return submodules_.at(submodule_order_[index]);
    }

    // Iterators
    auto begin() { return submodule_order_.begin(); }
    auto end() { return submodule_order_.end(); }
    auto begin() const { return submodule_order_.begin(); }
    auto end() const { return submodule_order_.end(); }

    // Forward is not typically called on ModuleList directly
    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("ModuleList is not meant to be called directly");
    }
};

// ============================================================================
// ModuleDict - Holds submodules in a dictionary
// ============================================================================

class ModuleDict : public Module {
public:
    ModuleDict() : Module("ModuleDict") {}

    explicit ModuleDict(
        std::initializer_list<std::pair<std::string, ModulePtr>> modules
    ) : Module("ModuleDict") {
        for (const auto& [name, module] : modules) {
            insert(name, module);
        }
    }

    // Insert or update
    void insert(const std::string& key, ModulePtr module) {
        // Check if already exists
        auto it = std::find(submodule_order_.begin(), submodule_order_.end(), key);
        if (it == submodule_order_.end()) {
            register_module(key, std::move(module));
        } else {
            submodules_[key] = std::move(module);
        }
    }

    // Update (same as insert)
    void update(const std::string& key, ModulePtr module) {
        insert(key, std::move(module));
    }

    // Remove
    void pop(const std::string& key) {
        auto it = std::find(submodule_order_.begin(), submodule_order_.end(), key);
        if (it != submodule_order_.end()) {
            submodule_order_.erase(it);
            submodules_.erase(key);
        }
    }

    // Clear all
    void clear() {
        submodule_order_.clear();
        submodules_.clear();
    }

    // Get size
    size_t size() const {
        return submodule_order_.size();
    }

    // Check if empty
    bool empty() const {
        return submodule_order_.empty();
    }

    // Check if key exists
    bool contains(const std::string& key) const {
        return submodules_.find(key) != submodules_.end();
    }

    // Access by key
    ModulePtr operator[](const std::string& key) {
        auto it = submodules_.find(key);
        if (it == submodules_.end()) {
            throw std::out_of_range("ModuleDict key not found: " + key);
        }
        return it->second;
    }

    // Access by key (alias)
    ModulePtr at(const std::string& key) {
        return (*this)[key];
    }

    // Get keys
    std::vector<std::string> keys() const {
        return submodule_order_;
    }

    // Get values
    std::vector<ModulePtr> values() {
        std::vector<ModulePtr> result;
        for (const auto& key : submodule_order_) {
            result.push_back(submodules_[key]);
        }
        return result;
    }

    // Get items
    std::vector<std::pair<std::string, ModulePtr>> items() {
        std::vector<std::pair<std::string, ModulePtr>> result;
        for (const auto& key : submodule_order_) {
            result.emplace_back(key, submodules_[key]);
        }
        return result;
    }

    // Forward is not typically called on ModuleDict directly
    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("ModuleDict is not meant to be called directly");
    }
};

// ============================================================================
// ParameterList - Holds parameters in a list
// ============================================================================

class ParameterList : public Module {
public:
    ParameterList() : Module("ParameterList") {}

    explicit ParameterList(std::initializer_list<Parameter> params)
        : Module("ParameterList") {
        for (const auto& param : params) {
            append(param);
        }
    }

    void append(Parameter param) {
        std::string name = std::to_string(size());
        register_parameter(name, std::move(param));
    }

    size_t size() const {
        return parameter_order_.size();
    }

    Parameter& operator[](size_t index) {
        if (index >= parameter_order_.size()) {
            throw std::out_of_range("ParameterList index out of range");
        }
        return parameters_[parameter_order_[index]];
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("ParameterList is not meant to be called directly");
    }
};

// ============================================================================
// ParameterDict - Holds parameters in a dictionary
// ============================================================================

class ParameterDict : public Module {
public:
    ParameterDict() : Module("ParameterDict") {}

    explicit ParameterDict(
        std::initializer_list<std::pair<std::string, Parameter>> params
    ) : Module("ParameterDict") {
        for (const auto& [name, param] : params) {
            insert(name, param);
        }
    }

    void insert(const std::string& key, Parameter param) {
        auto it = std::find(parameter_order_.begin(), parameter_order_.end(), key);
        if (it == parameter_order_.end()) {
            register_parameter(key, std::move(param));
        } else {
            parameters_[key] = std::move(param);
        }
    }

    size_t size() const {
        return parameter_order_.size();
    }

    bool contains(const std::string& key) const {
        return parameters_.find(key) != parameters_.end();
    }

    Parameter& operator[](const std::string& key) {
        auto it = parameters_.find(key);
        if (it == parameters_.end()) {
            throw std::out_of_range("ParameterDict key not found: " + key);
        }
        return it->second;
    }

    Tensor forward(const Tensor& input) override {
        throw std::runtime_error("ParameterDict is not meant to be called directly");
    }
};

} // namespace nn
} // namespace torch
