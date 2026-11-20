// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

// Cross-extension ABI. Execution must not call Python, touch Python reference
// counts, or acquire the GIL. The capsule owner retains all buffers until stop.
struct MoeNativeTask {
  static constexpr const char* capsule_name = "vllm_ascend.native_task.v1";
  uint64_t abi = 1;
  void* context = nullptr;
  void (*run)(void*) = nullptr;
};
