// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#ifndef _WIN32
#include <dlfcn.h>
#endif

// MSTX C ABI: no py::object creation, Python callback, or GIL on the worker.
class NativeRanges {
 public:
  NativeRanges() {
#ifdef _WIN32
    throw std::runtime_error("native MSTX requires Linux");
#else
    library_ = dlopen("libms_tools_ext.so", RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error("cannot load libms_tools_ext.so for native profiling");
    start_ = reinterpret_cast<Start>(dlsym(library_, "mstxRangeStartA"));
    end_ = reinterpret_cast<End>(dlsym(library_, "mstxRangeEnd"));
    if (!start_ || !end_) {
      dlclose(library_); library_ = nullptr;
      throw std::runtime_error("missing native MSTX range symbols");
    }
#endif
  }
  ~NativeRanges() {
#ifndef _WIN32
    if (library_) dlclose(library_);
#endif
  }
  void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_ || inflight_) throw std::invalid_argument("native measurement already active");
    count_ = errors_ = 0;
    active_ = true;
  }
  std::pair<uint64_t, uint64_t> stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    active_ = false;
    if (!drained_.wait_for(lock, std::chrono::seconds(30), [this] { return inflight_ == 0; }))
      throw std::invalid_argument("native measurement did not drain within 30s");
    return {count_, errors_};
  }
  uint64_t enter() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) return 0;
    ++inflight_;
    return ++count_;
  }
  void leave() {
    std::lock_guard<std::mutex> lock(mutex_);
    --inflight_;
    drained_.notify_all();
  }
  uint64_t begin(const std::string& name) {
    const auto id = start_(name.c_str(), nullptr);
    if (!id) { std::lock_guard<std::mutex> lock(mutex_); ++errors_; }
    return id;
  }
  void end(uint64_t id) { if (id) end_(id); }
 private:
  using Start = uint64_t (*)(const char*, void*);
  using End = void (*)(uint64_t);
  void* library_ = nullptr;
  Start start_ = nullptr;
  End end_ = nullptr;
  std::mutex mutex_;
  std::condition_variable drained_;
  bool active_ = false;
  uint64_t count_ = 0, errors_ = 0, inflight_ = 0;
};

class NativeRange {
 public:
  NativeRange(NativeRanges* profiler, const char* stage, const std::string& context)
      : profiler_(profiler), id_(profiler ? profiler->begin(std::string("moe_offload/") + stage + "|" + context) : 0) {}
  ~NativeRange() { if (profiler_) profiler_->end(id_); }
 private:
  NativeRanges* profiler_;
  uint64_t id_;
};

class NativeMeasurement {
 public:
  explicit NativeMeasurement(const std::shared_ptr<NativeRanges>& profiler)
      : profiler_(profiler.get()), call(profiler_ ? profiler_->enter() : 0) {}
  ~NativeMeasurement() { if (call) profiler_->leave(); }
  NativeRanges* get() const { return call ? profiler_ : nullptr; }
 private:
  NativeRanges* profiler_;
 public:
  const uint64_t call;
};
