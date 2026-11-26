// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <thread>

// The original Session thread plans requests. This second thread ONLY submits
// prepared DMA jobs. Neither thread waits for device transfer completion.
class NativePipeline;
struct PipelineTask {
  MoeNativeTask task;
  py::object owner;  // Created/destroyed with the GIL, never touched by workers.
  NativeUpdate* update;
  std::shared_ptr<NativePipeline> engine;
  uintptr_t request, device_ready, host_ready, mapping_device;
  size_t index;
  int dependency;
  uint64_t admitted = 0;
  PipelineTask(py::object capsule, std::shared_ptr<NativePipeline> e, uintptr_t r,
               uintptr_t d, uintptr_t h, uintptr_t mapping, int after);
  ~PipelineTask();
  void run();
};

class NativePipeline {
 public:
  struct Job {
    PipelineTask* task;
    uint64_t generation;
    NativeUpdate::Plan plan;
    std::shared_ptr<NativeMeasurement> measurement;
    std::string context;
  };
  explicit NativePipeline(const std::string& library, bool cpu_test) : cpu_(cpu_test) {
    if (!cpu_) {
#ifdef _WIN32
      throw std::runtime_error("native pipeline requires Linux/ACL");
#else
      acl_ = dlopen("libascendcl.so", RTLD_NOW | RTLD_LOCAL);
      kernel_ = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (!acl_ || !kernel_) { unload(); throw std::runtime_error("cannot load pipeline ACL/kernel library"); }
      auto get = reinterpret_cast<int (*)(void**)>(dlsym(acl_, "aclrtGetCurrentContext"));
      set_ = reinterpret_cast<int (*)(void*)>(dlsym(acl_, "aclrtSetCurrentContext"));
      complete_ = reinterpret_cast<void (*)(void*, void*, uint64_t)>(dlsym(kernel_, "host_sync_pipeline_complete_launch"));
      if (!get || !set_ || !complete_ || get(&context_) || !context_) {
        unload(); throw std::runtime_error("pipeline needs current ACL context and completion kernel; rebuild handshake");
      }
#endif
    }
    worker_ = std::thread([this] { loop(); });
  }
  ~NativePipeline() { close(); unload(); }
  size_t add(PipelineTask* task) {
    std::lock_guard<std::mutex> guard(mutex_);
    check();
    if (closed_ || task->dependency < -1 || task->dependency >= static_cast<int>(channels_.size()))
      throw std::invalid_argument("pipeline dependency must be an earlier registered demand");
    if (task->dependency >= 0 && (channels_[task->dependency]->update->prefetch || !task->update->prefetch))
      throw std::invalid_argument("pipeline dependency must connect demand to prefetch");
    channels_.push_back(task);
    submitted_.push_back(0);
    last_request_.push_back(0);
    request_time_.push_back(std::chrono::steady_clock::now());
    return channels_.size() - 1;
  }
  void enqueue(Job job) {
    std::lock_guard<std::mutex> guard(mutex_);
    check();
    if (closed_) throw std::runtime_error("pipeline closed");
    queue_.push_back(std::move(job));
    wake_.notify_one();
  }
  void check_progress(double timeout) {
    std::lock_guard<std::mutex> guard(mutex_);
    check();
    if (closed_) return;
    auto now = std::chrono::steady_clock::now();
    for (size_t i = 0; i < channels_.size(); ++i) {
      auto* c = channels_[i];
      uint64_t requested = read(c->request), completed = read(c->host_ready);
      if (requested != last_request_[i]) { last_request_[i] = requested; request_time_[i] = now; }
      if (requested != completed && std::chrono::duration<double>(now - request_time_[i]).count() > timeout)
        throw std::runtime_error("pipeline DMA completion timed out at channel " + std::to_string(i));
    }
  }
  void close() {
    { std::lock_guard<std::mutex> guard(mutex_); closed_ = true; paused_ = false; wake_.notify_all(); }
    if (worker_.joinable()) worker_.join();
  }
  void pause_for_test(bool pause) {
    if (!cpu_) throw std::invalid_argument("pause_for_test is CPU-only");
    std::lock_guard<std::mutex> guard(mutex_); paused_ = pause; wake_.notify_all();
  }
  uint64_t submitted_count() {
    std::lock_guard<std::mutex> guard(mutex_); check(); return jobs_;
  }
  static uint64_t read(uintptr_t address) {
    return reinterpret_cast<std::atomic<uint64_t>*>(address)->load(std::memory_order_acquire);
  }
 private:
  bool cpu_, closed_ = false, paused_ = false;
  void* acl_ = nullptr;
  void* kernel_ = nullptr;
  void* context_ = nullptr;
  int (*set_)(void*) = nullptr;
  void (*complete_)(void*, void*, uint64_t) = nullptr;
  std::thread worker_;
  std::mutex mutex_;
  std::condition_variable wake_;
  std::deque<Job> queue_;
  std::vector<PipelineTask*> channels_;
  std::vector<uint64_t> submitted_, last_request_;
  std::vector<std::chrono::steady_clock::time_point> request_time_;
  uint64_t jobs_ = 0;
  std::string error_;
  void check() { if (!error_.empty()) throw std::runtime_error(error_); }
  void unload() {
#ifndef _WIN32
    if (kernel_) dlclose(kernel_);
    if (acl_) dlclose(acl_);
#endif
    kernel_ = acl_ = nullptr;
  }
  size_t ready() {
    if (paused_) return queue_.size();
    for (size_t i = 0; i < queue_.size(); ++i) {
      int dependency = queue_[i].task->dependency;
      if (dependency < 0 || submitted_[dependency] == queue_[i].generation) return i;
    }
    return queue_.size();
  }
  void loop() {
    try {
      if (!cpu_ && set_(context_)) throw std::runtime_error("pipeline DMA ACL context failed");
      for (;;) {
        Job job;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          wake_.wait(lock, [this] { return closed_ || ready() < queue_.size(); });
          size_t i = ready();
          if (i == queue_.size()) {
            if (!queue_.empty()) throw std::runtime_error("pipeline closed with an unsatisfied dependency");
            break;
          }
          job = std::move(queue_[i]); queue_.erase(queue_.begin() + i);
        }
        auto& task = *job.task;
        auto& update = *task.update;
        auto& transfer = *update.transfer;
        NativeRange range(job.measurement->get(), "pipeline_dma_submit", job.context);
        transfer.enqueue_impl(job.plan.loads, job.measurement->get(), job.context);
        transfer.copy_mapping(task.mapping_device, update.mapping.data,
                              update.mapping.shape[0] * update.mapping.itemsize);
        // A zero-load job uses this SAME FIFO. It cannot overtake queued loads.
        if (cpu_) {
          reinterpret_cast<std::atomic<uint64_t>*>(task.device_ready)->store(job.generation, std::memory_order_release);
        } else {
          complete_(transfer.stream, reinterpret_cast<void*>(task.device_ready), job.generation);
        }
        // Independent watchdog evidence; consumer waits on the DEVICE flag.
        transfer.copy_completion(task.host_ready, task.device_ready);
        {
          std::lock_guard<std::mutex> guard(mutex_);
          submitted_[task.index] = job.generation; ++jobs_; wake_.notify_all();
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> guard(mutex_);
      error_ = e.what(); closed_ = true; queue_.clear(); wake_.notify_all();
      if (!cpu_) {
        std::fprintf(stderr, "[MOE-PIPELINE FATAL] %s; consumers remain blocked\n", e.what());
        std::abort();
      }
    }
  }
};

inline PipelineTask::PipelineTask(py::object capsule, std::shared_ptr<NativePipeline> e,
    uintptr_t r, uintptr_t d, uintptr_t h, uintptr_t mapping, int after)
    : owner(std::move(capsule)), engine(std::move(e)), request(r), device_ready(d),
      host_ready(h), mapping_device(mapping), dependency(after) {
  auto cap = owner.cast<py::capsule>();
  if (!cap.name() || std::string(cap.name()) != MoeNativeTask::capsule_name || !r || !d || !h || !mapping || r % 8 || d % 8 || h % 8)
    throw std::invalid_argument("invalid pipeline task/counter addresses");
  update = static_cast<NativeUpdate*>(static_cast<MoeNativeTask*>(cap.get_pointer())->context);
  index = engine->add(this);
  task.context = this;
  task.run = [](void* ptr) { static_cast<PipelineTask*>(ptr)->run(); };
}

inline void PipelineTask::run() {
  uint64_t generation = NativePipeline::read(request);
  if (!generation || generation != admitted + 1) throw std::runtime_error("pipeline request sequence skipped/repeated");
  auto measurement = std::make_shared<NativeMeasurement>(update->transfer->profiler);
  const auto context = measurement->call ? "backend=cpp|layer=" + std::to_string(update->layer) +
      "|kind=" + (update->prefetch ? std::string("prefetch") : std::string("reactive")) +
      "|rows=" + std::to_string(update->ids.shape[0]) + "|call=" + std::to_string(measurement->call) + "|pipeline=1" : "";
  NativeRange host(measurement->get(), "host_callback", context);
  NativeRange inner(measurement->get(), "update_weights", context);
  auto plan = update->plan();
  // This is a per-capture-site pinned snapshot, NOT the visible device mapping.
  // It may be committed early; the device mapping is published after weights.
  update->commit(plan);
  NativeRange enqueue_range(measurement->get(), "pipeline_enqueue", context);
  engine->enqueue({this, generation, std::move(plan), measurement, context});
  admitted = generation;
}

inline PipelineTask::~PipelineTask() {
  // Drain jobs before releasing the NativeUpdate capsule and its source owners.
  // Normal shutdown already closes the engine before Session releases tasks.
  engine->close();
}

inline void bind_native_pipeline(py::module_& m) {
  py::class_<NativePipeline, std::shared_ptr<NativePipeline>>(m, "NativePipeline")
      .def(py::init<const std::string&, bool>())
      .def("close", &NativePipeline::close, py::call_guard<py::gil_scoped_release>())
      .def("check_progress", &NativePipeline::check_progress)
      .def("submitted_count", &NativePipeline::submitted_count)
      .def("pause_for_test", &NativePipeline::pause_for_test);
  m.def("native_pipeline_task", [](py::object update, std::shared_ptr<NativePipeline> engine,
      uintptr_t request, uintptr_t ready, uintptr_t completed, uintptr_t mapping, int dependency) {
    auto* value = new PipelineTask(std::move(update), std::move(engine), request, ready, completed, mapping, dependency);
    auto capsule = py::capsule(&value->task, MoeNativeTask::capsule_name, [](void* p) {
      delete static_cast<PipelineTask*>(static_cast<MoeNativeTask*>(p)->context);
    });
    return py::make_tuple(std::move(capsule), value->index);
  });
}
