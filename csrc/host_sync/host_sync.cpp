// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "host_memory.h"
#include "protocol.h"

namespace py = pybind11;

namespace host_sync {

class WaitTimeout : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Session {
 public:
  Session(size_t slots, const std::string& backend, int device_id, const std::string& acl_library)
      : slots_(slots), backend_(backend), device_id_(device_id) {
    if (slots == 0 || slots > kMaxSlots) {
      throw std::invalid_argument("slots must be in [1, 4096]");
    }
    memory_ = std::make_unique<HostMemory>(slots * sizeof(Slot), backend, device_id, acl_library);
    allocation_bytes_ = memory_->size();
    signals_ = static_cast<Slot*>(memory_->data());
    for (size_t i = 0; i < slots_; ++i) new (&signals_[i]) Slot();
    callbacks_.resize(slots_);
    errors_.resize(slots_);
  }

  void configure_polling(const std::string& idle_mode, bool timing) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    if (state_ != "created") throw std::runtime_error("configure_polling requires an unstarted session");
    if (idle_mode != "yield" && idle_mode != "spin") throw std::invalid_argument("idle_mode must be yield or spin");
    idle_yield_ = idle_mode == "yield";
    poll_timing_ = timing;
    if (timing) {
      task_samples_.reserve(kMaxTimingSamples);
      idle_samples_.reserve(kMaxTimingSamples);
    }
  }

  py::dict polling_stats() {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    if (!finished_) throw std::runtime_error("stop the worker before reading polling_stats");
    py::dict result;
    result["idle_mode"] = idle_yield_ ? "yield" : "spin";
    result["timing_enabled"] = poll_timing_;
    result["idle_sample_stride"] = kIdleSampleStride;
    result["task_samples_dropped"] = task_samples_dropped_;
    result["idle_samples_dropped"] = idle_samples_dropped_;
    result["task_columns"] = py::make_tuple("slot", "generation", "claim_lock_wait_ns", "found_to_gil_ns",
        "gil_wait_ns", "callback_ns", "gil_release_ns", "completion_lock_wait_ns", "completion_publish_ns");
    result["idle_columns"] = py::make_tuple("scan_ns", "scan_lock_wait_ns", "idle_action_ns");
    result["task_samples"] = task_samples_;
    result["idle_samples"] = idle_samples_;
    return result;
  }

  // The Python facade owns an atexit reference until successful close().
  // This is only a last-resort cleanup for direct native users.
  ~Session() {
    stop_requested_.store(true, std::memory_order_release);
    if (worker_.joinable()) {
      // Worker never owns the Session; Python facade prevents last-reference
      // destruction in a callback. Never detach a thread using this object.
      if (worker_.get_id() == std::this_thread::get_id()) std::terminate();
      if (PyGILState_Check()) {
        py::gil_scoped_release release;
        worker_.join();
      } else {
        worker_.join();
      }
    }
  }

  void register_task(size_t slot, py::function callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    check_slot(slot);
    if (state_ != "created") {
      throw std::runtime_error("register_task is allowed only before start()");
    }
    if (callbacks_[slot]) {
      throw std::runtime_error("a callback is already registered for this slot");
    }
    callbacks_[slot] = std::move(callback);
  }

  void start(int64_t timeout_ms) {
    validate_timeout(timeout_ms);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      require_open();
      if (state_ != "created") {
        throw std::runtime_error("start() is allowed only once");
      }
      bool any_registered = false;
      for (const auto& callback : callbacks_) any_registered |= bool(callback);
      if (!any_registered) throw std::runtime_error("register a task before start()");
      state_ = "starting";
      try {
        worker_ = std::thread(&Session::run, this);
      } catch (...) {
        state_ = "stopped";
        finished_ = true;
        throw;
      }
    }
    {
      py::gil_scoped_release release;
      std::unique_lock<std::mutex> lock(mutex_);
      if (!wait_condition(lock, timeout_ms, [this] { return ready_ || finished_; })) {
        stop_requested_.store(true, std::memory_order_release);
        state_ = "stopping";
        throw WaitTimeout("worker startup timed out; call close() again to finish cleanup");
      }
      if (!startup_error_.empty()) throw std::runtime_error(startup_error_);
      if (state_ != "running") throw std::runtime_error("worker was stopped during startup");
    }
  }

  // Test producer only. No device kernels may access this session concurrently.
  void submit_for_test(size_t slot, uint64_t generation) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    reject_worker_call("submit_for_test");
    check_slot(slot);
    if (state_ != "running") throw std::runtime_error("session is not running");
    if (device_access_) throw std::runtime_error("device producer owns the session; Host test submission is disabled");
    if (!callbacks_[slot]) throw std::runtime_error("slot has no registered callback");
    auto& signal = signals_[slot];
    const uint64_t previous = signal.request.generation.load(std::memory_order_acquire);
    if (previous == std::numeric_limits<uint64_t>::max() || generation == 0 || generation != previous + 1) {
      throw std::invalid_argument("generation must be exactly previous request + 1, starting at 1");
    }
    if (previous != 0 && signal.completion.done_generation.load(std::memory_order_acquire) != previous &&
        signal.completion.error_generation.load(std::memory_order_acquire) != previous) {
      throw std::runtime_error("slot still has an outstanding request");
    }
    errors_[slot].clear();
    signal.completion.status.store(static_cast<uint64_t>(TaskStatus::Idle), std::memory_order_relaxed);
    signal.request.generation.store(generation, std::memory_order_release);
  }

  py::dict wait(size_t slot, uint64_t generation, int64_t timeout_ms) {
    validate_timeout(timeout_ms);
    {
      py::gil_scoped_release release;
      std::unique_lock<std::mutex> lock(mutex_);
      require_open();
      reject_worker_call("wait");
      check_slot(slot);
      if (generation == 0 || signals_[slot].request.generation.load(std::memory_order_acquire) != generation) {
        throw std::invalid_argument("wait requires the latest submitted generation");
      }
      const bool ready = wait_condition(lock, timeout_ms,
                                        [&] { return state_ == "closed" || terminal(slot, generation) || finished_; });
      if (!ready) throw WaitTimeout("task wait timed out; task is still owned by the session");
      require_open();
      if (signals_[slot].request.generation.load(std::memory_order_acquire) != generation) {
        throw std::runtime_error("slot was reused before wait() consumed its result");
      }
      if (signals_[slot].completion.error_generation.load(std::memory_order_acquire) == generation) {
        throw std::runtime_error(errors_[slot]);
      }
      if (signals_[slot].completion.done_generation.load(std::memory_order_acquire) != generation) {
        throw std::runtime_error("worker stopped before task completion");
      }
    }
    // Return a fresh snapshot with the GIL held, after releasing the mutex.
    return snapshot(slot);
  }

  py::dict snapshot(size_t slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    check_slot(slot);
    const auto& signal = signals_[slot];
    py::dict result;
    result["request_generation"] = signal.request.generation.load(std::memory_order_acquire);
    result["done_generation"] = signal.completion.done_generation.load(std::memory_order_acquire);
    result["error_generation"] = signal.completion.error_generation.load(std::memory_order_acquire);
    result["status"] = signal.completion.status.load(std::memory_order_acquire);
    result["error"] = errors_[slot];
    return result;
  }

  py::dict info() {
    std::lock_guard<std::mutex> lock(mutex_);
    py::dict result;
    result["backend"] = backend_;
    result["device_id"] = device_id_;
    result["slots"] = slots_;
    result["state"] = state_;
    result["protocol_version"] = kProtocolVersion;
    result["slot_stride"] = sizeof(Slot);
    result["request_offset"] = offsetof(Slot, request);
    result["done_offset"] = offsetof(Slot, completion) + offsetof(CompletionSignal, done_generation);
    result["error_offset"] = offsetof(Slot, completion) + offsetof(CompletionSignal, error_generation);
    result["status_offset"] = offsetof(Slot, completion) + offsetof(CompletionSignal, status);
    result["allocation_bytes"] = allocation_bytes_;
    result["host_address"] = reinterpret_cast<uintptr_t>(memory_->data());
    result["device_address"] = memory_->device_address();
    result["worker_thread_id"] = worker_id_ == std::thread::id() ? size_t(0) : std::hash<std::thread::id>{}(worker_id_);
    result["device_protocol_verified"] = false;
    result["device_access_active"] = device_access_;
    return result;
  }

  // Experimental device tests must keep this lease until the submitting
  // stream has synchronized successfully. It blocks Host producers/cleanup.
  uintptr_t begin_device_access() {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    reject_worker_call("begin_device_access");
    if (backend_ != "acl" || state_ != "running" || device_access_) {
      throw std::runtime_error("device access requires a running ACL session without an active lease");
    }
    for (size_t i = 0; i < slots_; ++i) {
      if (!callbacks_[i] || signals_[i].request.generation.load(std::memory_order_acquire) != 0) {
        throw std::runtime_error("device access requires registered, unused slots");
      }
    }
    device_access_ = true;
    return memory_->device_address();
  }

  void end_device_access(bool stream_synchronized) {
    std::lock_guard<std::mutex> lock(mutex_);
    require_open();
    reject_worker_call("end_device_access");
    if (!device_access_ || !stream_synchronized) {
      throw std::runtime_error("release requires an active lease and successful device stream synchronization");
    }
    device_access_ = false;
  }

  void stop(int64_t timeout_ms) {
    validate_timeout(timeout_ms);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      reject_worker_call("stop");
      if (state_ == "closed") return;
      if (device_access_)
        throw std::runtime_error("synchronize device users and release device access before stop/close");
      if (state_ == "created") {
        state_ = "stopped";
        finished_ = true;
        return;
      }
      stop_requested_.store(true, std::memory_order_release);
      if (!finished_) state_ = "stopping";
    }
    // Release the GIL before waiting or serializing joins: a callback can need
    // it to finish. On timeout we retain the thread, memory and callbacks.
    {
      py::gil_scoped_release release;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!wait_condition(lock, timeout_ms, [this] { return finished_; })) {
          throw WaitTimeout("stop timed out; callback is still running, resources retained");
        }
      }
      std::lock_guard<std::mutex> join_lock(join_mutex_);
      if (worker_.joinable()) worker_.join();
    }
  }

  void close(int64_t timeout_ms) {
    stop(timeout_ms);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (state_ == "closed") return;
      // Stage 1 has no device users. A future device backend must quiesce all
      // graph/stream users before it reaches this point.
      memory_->close();
      signals_ = nullptr;
      state_ = "closed";
    }
    // Decref outside the mutex; Python destructors may re-enter the facade.
    std::vector<py::function> retired;
    retired.swap(callbacks_);
    changed_.notify_all();
  }

 private:
  void require_open() const {
    if (state_ == "closed") throw std::runtime_error("session is closed");
  }

  void check_slot(size_t slot) const {
    if (slot >= slots_) throw std::out_of_range("slot index is out of range");
  }

  void reject_worker_call(const char* method) const {
    if (!finished_ && worker_id_ != std::thread::id() && worker_id_ == std::this_thread::get_id()) {
      throw std::runtime_error(std::string(method) + " cannot be called from a callback");
    }
  }

  static void validate_timeout(int64_t timeout_ms) {
    if (timeout_ms < -1) throw std::invalid_argument("timeout_ms must be >= 0, or -1 for no deadline");
  }

  template <typename Predicate>
  bool wait_condition(std::unique_lock<std::mutex>& lock, int64_t timeout_ms, Predicate predicate) {
    if (timeout_ms == -1) {
      changed_.wait(lock, predicate);
      return true;
    }
    return changed_.wait_for(lock, std::chrono::milliseconds(timeout_ms), predicate);
  }

  bool terminal(size_t slot, uint64_t generation) const {
    return signals_[slot].completion.done_generation.load(std::memory_order_acquire) == generation ||
           signals_[slot].completion.error_generation.load(std::memory_order_acquire) == generation;
  }

  void publish_error(size_t slot, uint64_t generation, const std::string& error, TaskStatus status) {
    errors_[slot] = error.substr(0, kMaxErrorLength);
    signals_[slot].completion.status.store(static_cast<uint64_t>(status), std::memory_order_relaxed);
    signals_[slot].completion.error_generation.store(generation, std::memory_order_release);
  }

  void run() noexcept {
    try {
      memory_->enter_worker_context();
      {
        std::lock_guard<std::mutex> lock(mutex_);
        worker_id_ = std::this_thread::get_id();
        ready_ = true;
        if (!stop_requested_.load(std::memory_order_acquire)) state_ = "running";
      }
      changed_.notify_all();
      poll();
    } catch (const std::exception& error) {
      std::lock_guard<std::mutex> lock(mutex_);
      startup_error_ = std::string("host worker failed: ") + error.what();
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      startup_error_ = "host worker failed with an unknown native exception";
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t slot = 0; slot < slots_; ++slot) {
        const uint64_t generation = signals_[slot].request.generation.load(std::memory_order_acquire);
        if (generation != 0 && !terminal(slot, generation)) {
          publish_error(slot, generation, "task cancelled because worker stopped", TaskStatus::Cancelled);
        }
      }
      state_ = "stopped";
      finished_ = true;
    }
    changed_.notify_all();
  }

  void poll() {
    uint64_t sweep = 0;
    while (!stop_requested_.load(std::memory_order_acquire)) {
      bool did_work = false;
      const bool sample_idle = poll_timing_ && (++sweep % kIdleSampleStride == 0);
      const uint64_t scan_start = sample_idle ? clock_ns() : 0;
      uint64_t scan_lock_wait = 0;
      for (size_t slot = 0; slot < slots_; ++slot) {
        uint64_t generation;
        const uint64_t claim_begin = poll_timing_ ? clock_ns() : 0;
        uint64_t claim_acquired = 0;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (poll_timing_) claim_acquired = clock_ns();
          if (sample_idle) scan_lock_wait += claim_acquired - claim_begin;
          if (stop_requested_.load(std::memory_order_acquire)) return;
          generation = signals_[slot].request.generation.load(std::memory_order_acquire);
          if (generation == 0 || terminal(slot, generation)) continue;
          signals_[slot].completion.status.store(static_cast<uint64_t>(TaskStatus::Running), std::memory_order_release);
        }
        const uint64_t found = poll_timing_ ? clock_ns() : 0;
        did_work = true;
        std::string failure;
        const uint64_t gil_begin = poll_timing_ ? clock_ns() : 0;
        uint64_t gil_acquired = 0, callback_end = 0;
        {
          py::gil_scoped_acquire acquire;
          if (poll_timing_) gil_acquired = clock_ns();
          try {
            if (!callbacks_[slot]) throw std::runtime_error("slot has no registered callback");
            callbacks_[slot]();
          } catch (const py::error_already_set& error) {
            failure = error.what();
          } catch (const std::exception& error) {
            failure = error.what();
          } catch (...) {
            failure = "unknown callback exception";
          }
          if (poll_timing_) callback_end = clock_ns();
        }
        const uint64_t released = poll_timing_ ? clock_ns() : 0;
        uint64_t completion_acquired = 0, published = 0;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (poll_timing_) completion_acquired = clock_ns();
          if (failure.empty()) {
            signals_[slot].completion.status.store(static_cast<uint64_t>(TaskStatus::Success),
                                                   std::memory_order_relaxed);
            signals_[slot].completion.done_generation.store(generation, std::memory_order_release);
          } else {
            publish_error(slot, generation, failure, TaskStatus::Error);
          }
          if (poll_timing_) published = clock_ns();
        }
        changed_.notify_all();
        if (poll_timing_) {
          if (task_samples_.size() < kMaxTimingSamples) {
            task_samples_.push_back({slot, generation, claim_acquired - claim_begin, gil_begin - found,
                gil_acquired - gil_begin, callback_end - gil_acquired, released - callback_end,
                completion_acquired - released, published - completion_acquired});
          } else {
            ++task_samples_dropped_;
          }
        }
      }
      if (!did_work) {
        const uint64_t before_idle = sample_idle ? clock_ns() : 0;
        if (idle_yield_) std::this_thread::yield();
        if (sample_idle) {
          const uint64_t after_idle = clock_ns();
          if (idle_samples_.size() < kMaxTimingSamples) {
            idle_samples_.push_back({before_idle - scan_start, scan_lock_wait, after_idle - before_idle});
          } else {
            ++idle_samples_dropped_;
          }
        }
      }
    }
  }

  static uint64_t clock_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  static constexpr size_t kMaxTimingSamples = 8192;
  static constexpr uint64_t kIdleSampleStride = 1024;
  bool idle_yield_ = true;
  bool poll_timing_ = false;
  std::vector<std::array<uint64_t, 9>> task_samples_;
  std::vector<std::array<uint64_t, 3>> idle_samples_;
  uint64_t task_samples_dropped_ = 0;
  uint64_t idle_samples_dropped_ = 0;
  size_t slots_;
  std::string backend_;
  int device_id_;
  size_t allocation_bytes_ = 0;
  std::unique_ptr<HostMemory> memory_;
  Slot* signals_ = nullptr;
  std::vector<py::function> callbacks_;
  std::vector<std::string> errors_;
  std::mutex mutex_;
  std::mutex join_mutex_;
  std::condition_variable changed_;
  std::thread worker_;
  std::thread::id worker_id_;
  std::atomic<bool> stop_requested_{false};
  std::string state_ = "created";
  std::string startup_error_;
  bool ready_ = false;
  bool finished_ = false;
  bool device_access_ = false;
};

}  // namespace host_sync

PYBIND11_MODULE(_vllm_ascend_host_sync, module) {
  module.attr("API_VERSION") = 2;
  py::register_exception<host_sync::WaitTimeout>(module, "WaitTimeout", PyExc_TimeoutError);
  py::class_<host_sync::Session>(module, "_Session")
      .def(py::init<size_t, const std::string&, int, const std::string&>(), py::arg("slots"), py::arg("backend"),
           py::arg("device_id"), py::arg("acl_library"))
      .def("register_task", &host_sync::Session::register_task)
      .def("configure_polling", &host_sync::Session::configure_polling,
           py::arg("idle_mode") = "yield", py::arg("timing") = false)
      .def("polling_stats", &host_sync::Session::polling_stats)
      .def("start", &host_sync::Session::start, py::arg("timeout_ms") = 5000)
      .def("submit_for_test", &host_sync::Session::submit_for_test)
      .def("wait", &host_sync::Session::wait, py::arg("slot"), py::arg("generation"), py::arg("timeout_ms") = 5000)
      .def("snapshot", &host_sync::Session::snapshot)
      .def("info", &host_sync::Session::info)
      .def("begin_device_access", &host_sync::Session::begin_device_access)
      .def("end_device_access", &host_sync::Session::end_device_access, py::arg("stream_synchronized"))
      .def("stop", &host_sync::Session::stop, py::arg("timeout_ms") = 5000)
      .def("close", &host_sync::Session::close, py::arg("timeout_ms") = 5000);
}
