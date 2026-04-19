// ============================================================================
// deepspeed.h — DeepSpeed-style features built on top of PromeTorch's FSDP.
// ============================================================================
// This is the umbrella header bundling the three DeepSpeed-equivalent
// capabilities:
//
//   * CPU offload for optimizer states         (deepspeed_offload.h)
//       torch::distributed::deepspeed::OffloadOptimizer
//
//   * 1F1B pipeline schedule                   (pipeline_schedule.h)
//       torch::distributed::Pipeline1F1BStage
//       torch::distributed::PipelineScheduleConfig
//       torch::distributed::pipeline_1f1b_selftest_main()
//
//   * ZeRO-3 with prefetch + hierarchy         (zero3.h)
//       torch::distributed::ZeROStage3
//       torch::distributed::ZeROStage3Config
//
// Gradient checkpointing is already provided by torch/utils/checkpoint.h and
// is the recommended DeepSpeed-style activation-memory optimisation.
// Re-export it here for discoverability.
//
// All implementations use file-based collectives over /dev/shm so they run
// unchanged on Elbrus (LCC) and on Windows/MSVC.
// ============================================================================
#pragma once

#include "torch/distributed/fsdp.h"
#include "torch/distributed/deepspeed_offload.h"
#include "torch/distributed/pipeline_schedule.h"
#include "torch/distributed/zero3.h"
#include "torch/utils/checkpoint.h"

namespace torch { namespace distributed { namespace deepspeed {
// Convenience alias so callers can write
//     deepspeed::Pipeline1F1BStage ...
// without reaching into the parent namespace.
using torch::distributed::Pipeline1F1BStage;
using torch::distributed::PipelineScheduleConfig;
using torch::distributed::ZeROStage3;
using torch::distributed::ZeROStage3Config;
}}}  // namespace torch::distributed::deepspeed
