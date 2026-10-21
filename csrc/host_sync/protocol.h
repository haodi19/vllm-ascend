// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace host_sync {

// Experimental ABI v1. Device-side access and coherence are NOT implemented
// or certified in stage 1. Ordinary CPU tests use real acquire/release atomics.
// Separate the device-owned request from host-owned completion by 512 bytes.
constexpr size_t kSignalAlignment = 512;
constexpr uint64_t kProtocolVersion = 1;
constexpr size_t kMaxSlots = 4096;
constexpr size_t kMaxErrorLength = 4096;

enum class TaskStatus : uint64_t {
  Idle = 0,
  Running = 1,
  Success = 2,
  Error = 3,
  Cancelled = 4,
};

struct alignas(kSignalAlignment) RequestSignal {
  std::atomic<uint64_t> generation{0};
};

struct alignas(kSignalAlignment) CompletionSignal {
  std::atomic<uint64_t> done_generation{0};
  std::atomic<uint64_t> error_generation{0};
  std::atomic<uint64_t> status{static_cast<uint64_t>(TaskStatus::Idle)};
};

struct Slot {
  RequestSignal request;
  CompletionSignal completion;
};

static_assert(std::atomic<uint64_t>::is_always_lock_free, "host_sync requires lock-free 64-bit atomics");
static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t), "unexpected atomic representation");
static_assert(sizeof(RequestSignal) == kSignalAlignment);
static_assert(sizeof(CompletionSignal) == kSignalAlignment);
static_assert(sizeof(Slot) == 2 * kSignalAlignment);
static_assert(offsetof(Slot, completion) == kSignalAlignment);

}  // namespace host_sync
