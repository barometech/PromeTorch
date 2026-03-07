#pragma once

#include "torch/quantization/quantize.h"
#include "torch/quantization/observer.h"
#include "torch/nn/modules/quantized.h"
#include "torch/nn/module.h"
#include <memory>
#include <functional>
#include <iostream>

namespace torch {
namespace quantization {

using at::Tensor;

// ============================================================================
// Quantization Pipeline
// ============================================================================

// Configuration for quantization
struct QConfig {
    enum class Scheme { PER_TENSOR, PER_CHANNEL };
    enum class ObserverType { MIN_MAX, HISTOGRAM, PER_CHANNEL_MIN_MAX };

    Scheme scheme = Scheme::PER_TENSOR;
    ObserverType weight_observer = ObserverType::MIN_MAX;
    ObserverType activation_observer = ObserverType::MIN_MAX;
    int64_t quant_min = 0;
    int64_t quant_max = 255;
};

inline QConfig default_qconfig() {
    return QConfig{};
}

inline QConfig default_per_channel_qconfig() {
    QConfig cfg;
    cfg.scheme = QConfig::Scheme::PER_CHANNEL;
    cfg.weight_observer = QConfig::ObserverType::PER_CHANNEL_MIN_MAX;
    return cfg;
}

// ============================================================================
// Prepare — insert observers into model (records activation statistics)
// ============================================================================

struct PreparedModel {
    torch::nn::Module* model;
    std::vector<std::unique_ptr<Observer>> activation_observers;
    std::vector<std::unique_ptr<Observer>> weight_observers;
    QConfig qconfig;

    PreparedModel(torch::nn::Module* m, const QConfig& cfg = QConfig{})
        : model(m), qconfig(cfg) {}
};

inline PreparedModel prepare(torch::nn::Module& model, const QConfig& qconfig = QConfig{}) {
    PreparedModel prepared(&model, qconfig);

    // Create observers for each parameter
    for (auto& [name, param] : model.named_parameters()) {
        std::unique_ptr<Observer> obs;
        switch (qconfig.weight_observer) {
            case QConfig::ObserverType::HISTOGRAM:
                obs = std::make_unique<HistogramObserver>(2048, qconfig.quant_min, qconfig.quant_max);
                break;
            case QConfig::ObserverType::PER_CHANNEL_MIN_MAX:
                obs = std::make_unique<PerChannelMinMaxObserver>(0, qconfig.quant_min, qconfig.quant_max);
                break;
            default:
                obs = std::make_unique<MinMaxObserver>(qconfig.quant_min, qconfig.quant_max);
                break;
        }
        prepared.weight_observers.push_back(std::move(obs));
    }

    return prepared;
}

// ============================================================================
// Calibrate — run data through model to collect statistics
// ============================================================================

inline void calibrate(PreparedModel& prepared, const std::vector<Tensor>& calibration_data) {
    // Observe weights
    auto params = prepared.model->parameters();
    for (size_t i = 0; i < params.size() && i < prepared.weight_observers.size(); ++i) {
        if (params[i]->defined()) {
            prepared.weight_observers[i]->forward(params[i]->data());
        }
    }

    // Run calibration data through model to collect activation stats
    for (const auto& data : calibration_data) {
        prepared.model->forward(data);
    }

    std::cout << "[Quantization] Calibration complete with "
              << calibration_data.size() << " samples" << std::endl;
}

// ============================================================================
// Convert — replace float layers with quantized versions
// Returns quantization parameters for each layer
// ============================================================================

struct QuantizationResult {
    std::vector<std::pair<double, int64_t>> layer_params;  // (scale, zero_point) per layer
    bool success = true;
};

inline QuantizationResult convert(PreparedModel& prepared) {
    QuantizationResult result;

    auto params = prepared.model->parameters();

    for (size_t i = 0; i < params.size() && i < prepared.weight_observers.size(); ++i) {
        auto [scale, zero_point] = prepared.weight_observers[i]->calculate_qparams();
        result.layer_params.push_back({scale, zero_point});

        // Quantize and dequantize in-place (fake quantization for now)
        if (params[i]->defined()) {
            Tensor& param_tensor = params[i]->data();
            auto qt = quantize_per_tensor(param_tensor, scale, zero_point);
            Tensor dequantized = qt.dequantize();
            param_tensor.copy_(dequantized);
        }
    }

    std::cout << "[Quantization] Converted " << result.layer_params.size()
              << " parameter tensors" << std::endl;

    return result;
}

// ============================================================================
// Convenience: one-shot quantization
// ============================================================================

inline QuantizationResult quantize_model(torch::nn::Module& model,
                                          const std::vector<Tensor>& calibration_data,
                                          const QConfig& qconfig = QConfig{}) {
    auto prepared = prepare(model, qconfig);
    calibrate(prepared, calibration_data);
    return convert(prepared);
}

} // namespace quantization
} // namespace torch
