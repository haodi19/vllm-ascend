// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBufferBytes = 32;
constexpr uint32_t kGenerationBytes = 8;
constexpr uint64_t kSuccess = 1;
constexpr uint64_t kCallbackError = 2;
constexpr uint64_t kPollLimit = 3;
constexpr uint64_t kInvalidGeneration = 4;

// MTE completion barriers order local scalar access and buffer reuse. Disabling
// L2 on mapped Host addresses does not by itself prove system-wide coherence;
// that is what the isolated on-device test checks on the target machine.
__aicore__ inline void Read(LocalTensor<uint8_t> local, GlobalTensor<uint8_t> global) {
  SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
  DataCopyExtParams copy{1, kBufferBytes, 0, 0, 0};
  DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
  DataCopyPad(local, global, copy, pad);
  SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
}

__aicore__ inline void Write(GlobalTensor<uint8_t> global, LocalTensor<uint8_t> local, uint32_t bytes) {
  SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
  WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
  DataCopyExtParams copy{1, bytes, 0, 0, 0};
  DataCopyPad(global, local, copy);
  SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}
}  // namespace

extern "C" __global__ __aicore__ void host_sync_publish(GM_ADDR request, GM_ADDR state) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<TPosition::VECCALC> scratch;
  pipe.InitBuffer(scratch, kBufferBytes);
  auto local = scratch.Get<uint8_t>();
  auto words = local.ReinterpretCast<uint64_t>();
  GlobalTensor<uint8_t> requestGm, stateGm;
  requestGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(request));
  stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(state));
  requestGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  stateGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  Read(local, stateGm);
  const uint64_t previous = words.GetValue(0);
  const uint64_t generation = previous + 1;
  words.SetValue(0, generation);
  words.SetValue(1, generation == 0 ? kInvalidGeneration : 0);
  words.SetValue(2, 0);
  words.SetValue(3, 0);
  Write(stateGm, local, kBufferBytes);
  if (generation != 0) Write(requestGm, local, kGenerationBytes);
}

extern "C" __global__ __aicore__ void host_sync_wait(GM_ADDR completion, GM_ADDR state, uint64_t maxPolls) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<TPosition::VECCALC> scratch;
  pipe.InitBuffer(scratch, kBufferBytes);
  auto local = scratch.Get<uint8_t>();
  auto words = local.ReinterpretCast<uint64_t>();
  GlobalTensor<uint8_t> completionGm, stateGm;
  completionGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(completion));
  stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(state));
  completionGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  stateGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  Read(local, stateGm);
  const uint64_t expected = words.GetValue(0);
  uint64_t code = expected == 0 ? kInvalidGeneration : kPollLimit;
  uint64_t observed = 0;
  uint64_t polls = 0;
  if (expected != 0 || maxPolls == 0) {
    for (; maxPolls == 0 || polls < maxPolls; ++polls) {
      Read(local, completionGm);
      observed = words.GetValue(0);
      if (words.GetValue(1) == expected && maxPolls != 0) {
        code = kCallbackError;
        ++polls;
        break;
      }
      if (expected != 0 && observed == expected && words.GetValue(1) != expected) {
        code = kSuccess;
        ++polls;
        break;
      }
    }
  }
  words.SetValue(0, expected);
  words.SetValue(1, code);
  words.SetValue(2, observed);
  words.SetValue(3, polls);
  Write(stateGm, local, kBufferBytes);
}

// Kernel-launch wrappers only; no Torch registration or system OPP install.
extern "C" void host_sync_publish_launch(void* stream, void* request, void* state) {
  host_sync_publish<<<1, nullptr, stream>>>(request, state);
}

extern "C" void host_sync_wait_launch(void* stream, void* completion, void* state, uint64_t maxPolls) {
  host_sync_wait<<<1, nullptr, stream>>>(completion, state, maxPolls);
}

// Inference must never resume on a poll limit or callback failure. The owning
// Host dispatcher terminates its engine process on failure/stalled generation.
extern "C" void host_sync_wait_strict_launch(void* stream, void* completion, void* state) {
  host_sync_wait<<<1, nullptr, stream>>>(completion, state, 0);
}

// Stage 5 consumer: check the handshake before consuming newly copied data.
// Output is [first_weight + mapping + route, last_weight, mapping, route].
extern "C" __global__ __aicore__ void host_sync_consume(GM_ADDR weights, GM_ADDR mapping, GM_ADDR route, GM_ADDR state,
                                                        GM_ADDR output, uint64_t count, uint64_t requireSuccess) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  TPipe pipe;
  TBuf<TPosition::VECCALC> scratch;
  pipe.InitBuffer(scratch, kBufferBytes);
  auto local = scratch.Get<uint8_t>();
  auto words = local.ReinterpretCast<uint64_t>();
  GlobalTensor<uint8_t> weightGm, mappingGm, routeGm, stateGm, outputGm;
  weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(weights));
  mappingGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(mapping));
  routeGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(route));
  stateGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(state));
  outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
  weightGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  mappingGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  routeGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  stateGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  if (requireSuccess != 0) {
    Read(local, stateGm);
    if (words.GetValue(1) != kSuccess) {
      for (uint32_t i = 0; i < 4; ++i) words.SetValue(i, 0);
      Write(outputGm, local, kBufferBytes);
      return;
    }
  }
  Read(local, routeGm);
  const uint64_t routeValue = words.GetValue(0);
  Read(local, mappingGm);
  const uint64_t mappingValue = words.GetValue(0);
  Read(local, weightGm);
  const uint64_t first = words.GetValue(0);
  Read(local, weightGm[(count - 4) * kGenerationBytes]);
  const uint64_t last = words.GetValue(3);
  words.SetValue(0, first + mappingValue + routeValue);
  words.SetValue(1, last);
  words.SetValue(2, mappingValue);
  words.SetValue(3, routeValue);
  Write(outputGm, local, kBufferBytes);
}

extern "C" void host_sync_consume_launch(void* stream, void* weights, void* mapping, void* route, void* state,
                                         void* output, uint64_t count, uint64_t requireSuccess) {
  host_sync_consume<<<1, nullptr, stream>>>(weights, mapping, route, state, output, count, requireSuccess);
}
