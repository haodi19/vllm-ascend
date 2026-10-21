// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>

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
    return result;
  }

  void stop(int64_t timeout_ms) {
    validate_timeout(timeout_ms);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      reject_worker_call("stop");
      if (state_ == "closed") return;
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
    while (!stop_requested_.load(std::memory_order_acquire)) {
      bool did_work = false;
      for (size_t slot = 0; slot < slots_; ++slot) {
        uint64_t generation;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (stop_requested_.load(std::memory_order_acquire)) return;
          generation = signals_[slot].request.generation.load(std::memory_order_acquire);
          if (generation == 0 || terminal(slot, generation)) continue;
          signals_[slot].completion.status.store(static_cast<uint64_t>(TaskStatus::Running), std::memory_order_release);
        }
        did_work = true;
        std::string failure;
        {
          py::gil_scoped_acquire acquire;
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
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (failure.empty()) {
            signals_[slot].completion.status.store(static_cast<uint64_t>(TaskStatus::Success),
                                                   std::memory_order_relaxed);
            signals_[slot].completion.done_generation.store(generation, std::memory_order_release);
          } else {
            publish_error(slot, generation, failure, TaskStatus::Error);
          }
        }
        changed_.notify_all();
      }
      // Stage 1 favors an inspectable, cooperative host loop. No spin/yield
      // tuning or performance claim is part of this foundation.
      if (!did_work) std::this_thread::yield();
    }
  }

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
};

}  // namespace host_sync

PYBIND11_MODULE(_vllm_ascend_host_sync, module) {
  module.attr("API_VERSION") = 1;
  py::register_exception<host_sync::WaitTimeout>(module, "WaitTimeout", PyExc_TimeoutError);
  py::class_<host_sync::Session>(module, "_Session")
      .def(py::init<size_t, const std::string&, int, const std::string&>(), py::arg("slots"), py::arg("backend"),
           py::arg("device_id"), py::arg("acl_library"))
      .def("register_task", &host_sync::Session::register_task)
      .def("start", &host_sync::Session::start, py::arg("timeout_ms") = 5000)
      .def("submit_for_test", &host_sync::Session::submit_for_test)
      .def("wait", &host_sync::Session::wait, py::arg("slot"), py::arg("generation"), py::arg("timeout_ms") = 5000)
      .def("snapshot", &host_sync::Session::snapshot)
      .def("info", &host_sync::Session::info)
      .def("stop", &host_sync::Session::stop, py::arg("timeout_ms") = 5000)
      .def("close", &host_sync::Session::close, py::arg("timeout_ms") = 5000);
}
