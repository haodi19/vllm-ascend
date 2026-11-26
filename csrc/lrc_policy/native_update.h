// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include "native_task.h"
#include "native_profile.h"
#ifndef _WIN32
#include <dlfcn.h>
#endif

// Included after Policy. Python is used only while registering descriptors.
// The replay path below owns no temporary Python/Torch objects.
class NativeTransfers {
 public:
  static constexpr int kHostToDevice = 1;
  static constexpr int kDeviceToHost = 2;
  static constexpr size_t kCompletionBytes = 32;
  static constexpr int32_t kCopyTimeoutMs = 30000;
  using Addresses = std::vector<std::vector<uintptr_t>>;
  Addresses sources, destinations;
  std::vector<size_t> sizes;
  void* stream;
  bool cpu;
  std::shared_ptr<NativeRanges> profiler;
  uint64_t batches = 0, copies = 0, bytes = 0;
  std::vector<std::pair<Id, Id>> identity_loads;
  NativeTransfers(Addresses src, Addresses dst, std::vector<size_t> lengths,
                  uintptr_t stream_ptr, const std::string& library, bool cpu_test)
      : sources(std::move(src)), destinations(std::move(dst)), sizes(std::move(lengths)),
        stream(reinterpret_cast<void*>(stream_ptr)), cpu(cpu_test) {
    if (sources.empty() || destinations.empty() || sizes.empty())
      throw std::invalid_argument("empty native transfer table");
    for (auto size : sizes) if (!size) throw std::invalid_argument("zero copy size");
    for (const auto* table : {&sources, &destinations})
      for (const auto& row : *table) {
        if (row.size() != sizes.size()) throw std::invalid_argument("copy component count mismatch");
        for (size_t part = 0; part < row.size(); ++part) {
          if (!row[part]) throw std::invalid_argument("null copy address");
          if (sizes[part] > std::numeric_limits<uintptr_t>::max() - row[part])
            throw std::invalid_argument("copy address range overflow");
        }
      }
    for (size_t i = 0; i < std::min(sources.size(), destinations.size()); ++i)
      identity_loads.emplace_back(i, i);
    if (cpu) return;
    if (!stream) throw std::invalid_argument("null ACL stream");
#ifdef _WIN32
    throw std::runtime_error("ACL native copies require Linux");
#else
    library_ = dlopen(library.empty() ? "libascendcl.so" : library.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error("cannot load libascendcl for native copies");
    try {
      copy_ = symbol<Copy>("aclrtMemcpyAsync");
      wait_ = symbol<Wait>("aclrtSynchronizeStreamWithTimeout");
    } catch (...) { dlclose(library_); library_ = nullptr; throw; }
#endif
  }
  ~NativeTransfers() {
#ifndef _WIN32
    if (library_) dlclose(library_);
#endif
  }
  // Python converts the small plan once at entry; this entire enqueue loop runs
  // without the GIL. Callers retain the owners and the existing completion gate.
  void enqueue(const std::vector<std::pair<Id, Id>>& loads) {
    // Validate the whole batch before any transfer, including a late bad entry.
    for (auto [slot, eid] : loads)
      if (slot < 0 || static_cast<size_t>(slot) >= destinations.size() ||
          eid < 0 || static_cast<size_t>(eid) >= sources.size())
        throw std::runtime_error("native copy expert/slot outside table");
    enqueue_impl(loads);
  }
  void enqueue_impl(const std::vector<std::pair<Id, Id>>& loads,
                    NativeRanges* ranges = nullptr, const std::string& context = "") {
    {
      NativeRange load_range(ranges, "load_submit", context);
      NativeRange copy_range(ranges, "copy_submit", context);
      for (auto [slot, eid] : loads) {
        if (slot < 0 || static_cast<size_t>(slot) >= destinations.size() ||
            eid < 0 || static_cast<size_t>(eid) >= sources.size())
          throw std::runtime_error("native copy expert/slot outside table");
        for (size_t part = 0; part < sizes.size(); ++part) {
          void* dst = reinterpret_cast<void*>(destinations[slot][part]);
          const void* src = reinterpret_cast<void*>(sources[eid][part]);
          if (cpu) std::memcpy(dst, src, sizes[part]);
          else check(copy_(dst, sizes[part], src, sizes[part], kHostToDevice, stream), "aclrtMemcpyAsync(H2D)");
          ++copies;
          bytes += sizes[part];
        }
      }
    }
  }
  void enqueue_all() {
    if (sources.size() != destinations.size())
      throw std::runtime_error("full native overwrite requires one slot per expert");
    enqueue(identity_loads);
  }
  void copy_mapping(uintptr_t destination, const void* source, size_t bytes) {
    if (cpu) std::memcpy(reinterpret_cast<void*>(destination), source, bytes);
    else check(copy_(reinterpret_cast<void*>(destination), bytes, source, bytes, kHostToDevice, stream), "pipeline mapping H2D");
  }
  void copy_completion(uintptr_t host, uintptr_t device) {
    if (cpu) {
      auto generation = reinterpret_cast<std::atomic<uint64_t>*>(device)->load(std::memory_order_acquire);
      reinterpret_cast<std::atomic<uint64_t>*>(host)->store(generation, std::memory_order_release);
    }
    else check(copy_(reinterpret_cast<void*>(host), kCompletionBytes, reinterpret_cast<void*>(device),
                     kCompletionBytes, kDeviceToHost, stream), "pipeline completion D2H");
  }
  void submit(const std::vector<std::pair<Id, Id>>& loads,
              NativeRanges* ranges = nullptr, const std::string& context = "") {
    // Keep the native graph baseline's validation order and measured ranges.
    enqueue_impl(loads, ranges, context);
    // Preserve the completion barrier even when this batch has no loads.
    NativeRange wait_range(ranges, "synchronize", context);
    if (!cpu) check(wait_(stream, kCopyTimeoutMs), "aclrtSynchronizeStreamWithTimeout");
    ++batches;
  }
 private:
  using Copy = int (*)(void*, size_t, const void*, size_t, int, void*);
  using Wait = int (*)(void*, int32_t);
  void* library_ = nullptr;
  Copy copy_ = nullptr;
  Wait wait_ = nullptr;
  static void check(int rc, const char* operation) {
    if (rc) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(rc));
  }
  template <typename T> T symbol(const char* name) {
#ifndef _WIN32
    auto p = dlsym(library_, name);
    if (p) return reinterpret_cast<T>(p);
#endif
    throw std::runtime_error(std::string("missing ACL symbol: ") + name);
  }
};

struct NativeArray {
  char* data = nullptr;
  std::vector<py::ssize_t> shape, strides;
  size_t itemsize = 0;
  explicit NativeArray(const py::array& array) {
    auto info = array.request();
    data = static_cast<char*>(info.ptr);
    shape = info.shape; strides = info.strides; itemsize = info.itemsize;
  }
  Id id(size_t r, size_t c) const {
    auto ptr = data + r * strides[0] + c * strides[1];
    return itemsize == 4 ? read_scalar<int32_t>(ptr) : read_scalar<int64_t>(ptr);
  }
  double score(size_t r, size_t c) const {
    auto ptr = data + r * strides[0] + c * strides[1];
    return itemsize == 4 ? read_scalar<float>(ptr) : read_scalar<double>(ptr);
  }
  Id mapping(size_t e) const {
    auto ptr = data + e * strides[0];
    return itemsize == 4 ? read_scalar<int32_t>(ptr) : read_scalar<int64_t>(ptr);
  }
  void write(size_t e, Id value) const {
    auto ptr = data + e * strides[0];
    if (itemsize == 4) { int32_t v = static_cast<int32_t>(value); std::memcpy(ptr, &v, 4); }
    else std::memcpy(ptr, &value, 8);
  }
};

class NativeUpdate {
 public:
  MoeNativeTask task;
  std::shared_ptr<Policy> policy;
  std::shared_ptr<NativeTransfers> transfer;
  // Owners are constructed/destroyed with the GIL, never touched by execute().
  py::array ids_owner, map_owner;
  py::object scores_owner, buffers_owner;
  NativeArray ids, mapping;
  std::unique_ptr<NativeArray> scores;
  Id layer, cap;
  bool prefetch;
  struct Plan {
    std::vector<std::pair<Id, Id>> loads;
    std::vector<Id> victims, missing;
  };

  NativeUpdate(std::shared_ptr<Policy> p, std::shared_ptr<NativeTransfers> t,
               Id layer_idx, py::array id_array, py::object score_array,
               py::array map_array, bool pf, Id limit, py::object owners)
      : policy(std::move(p)), transfer(std::move(t)), ids_owner(id_array), map_owner(map_array),
        scores_owner(score_array), buffers_owner(owners), ids(id_array), mapping(map_array),
        layer(layer_idx), cap(limit), prefetch(pf) {
    auto integer = [](const py::array& a) {
      return a.dtype().is(py::dtype::of<int32_t>()) || a.dtype().is(py::dtype::of<int64_t>());
    };
    if (ids.shape.size() != 2 || !integer(id_array) || mapping.shape.size() != 1 ||
        !integer(map_array) || !map_array.writeable() ||
        !(map_array.flags() & py::array::c_style) || cap < 1 ||
        mapping.shape[0] != policy->num_experts() ||
        mapping.shape[0] != static_cast<Id>(transfer->sources.size()))
      throw std::invalid_argument("invalid native IDs/mapping/cap/table");
    if (layer < 0 || layer >= policy->num_layers()) throw std::invalid_argument("invalid native layer");
    if (!score_array.is_none()) {
      auto arr = score_array.cast<py::array>();
      scores = std::make_unique<NativeArray>(arr);
      if (scores->shape != ids.shape ||
          !(arr.dtype().is(py::dtype::of<float>()) || arr.dtype().is(py::dtype::of<double>())))
        throw std::invalid_argument("native scores must match IDs and be float32/64");
    }
    task.context = this;
    task.run = [](void* self) { static_cast<NativeUpdate*>(self)->execute(); };
  }

  void execute() {
    NativeMeasurement measurement(transfer->profiler);
    const std::string context = measurement.call ?
        "backend=cpp|layer=" + std::to_string(layer) + "|kind=" +
        (prefetch ? "prefetch" : "reactive") + "|rows=" + std::to_string(ids.shape[0]) +
        "|call=" + std::to_string(measurement.call) : "";
    NativeRange host_range(measurement.get(), "host_callback", context);
    NativeRange update_range(measurement.get(), "update_weights", context);
    std::lock_guard<std::recursive_mutex> guard(policy->mutex);
    auto work = plan();
    transfer->submit(work.loads, measurement.get(), context);
    commit(work);
  }

  Plan plan() {
    std::lock_guard<std::recursive_mutex> guard(policy->mutex);
    const Id n = mapping.shape[0];
    std::vector<Id> resident, original(n);
    std::set<Id> occupied;
    for (Id e = 0; e < n; ++e) {
      original[e] = mapping.mapping(e);
      if (original[e] < -1 || original[e] >= static_cast<Id>(transfer->destinations.size()) ||
          (original[e] >= 0 && !occupied.insert(original[e]).second))
        throw std::runtime_error("native mapping has invalid or duplicate slots");
      if (original[e] >= 0) resident.push_back(e);
    }
    std::vector<std::vector<Id>> rows(ids.shape[0], std::vector<Id>(ids.shape[1]));
    std::vector<std::vector<double>> values;
    if (scores) values.assign(ids.shape[0], std::vector<double>(ids.shape[1]));
    std::unordered_set<Id> needed;
    std::vector<Id> candidates;
    std::vector<double> best(n, -std::numeric_limits<double>::infinity());
    for (Id r = 0; r < ids.shape[0]; ++r) for (Id c = 0; c < ids.shape[1]; ++c) {
      Id e = ids.id(r, c);
      rows[r][c] = e;
      double value = scores ? scores->score(r, c) : -static_cast<double>(c * ids.shape[0] + r);
      if (scores) values[r][c] = value;
      if (e < 0 || e >= n) {
        if (prefetch) throw std::runtime_error("invalid prefetch expert ID");
        continue;
      }
      needed.insert(e);
      if (original[e] < 0 && value > best[e]) {
        if (best[e] == -std::numeric_limits<double>::infinity()) candidates.push_back(e);
        best[e] = value;
      }
    }
    if (!prefetch) policy->observe_values(layer, rows, values, bool(scores));
    std::vector<Id> missing;
    if (prefetch) {
      std::stable_sort(candidates.begin(), candidates.end(), [&](Id a, Id b) { return best[a] > best[b]; });
      candidates.resize(std::min(static_cast<size_t>(cap), candidates.size()));
      missing = candidates;
    } else for (Id e : needed) if (original[e] < 0) missing.push_back(e);
    // Explicit deterministic ordering, independent of Python's hash table.
    // Expert selection and victim ordering are unchanged; physical slot pairing
    // can differ. The mapping is committed only after all transfers complete.
    std::sort(missing.begin(), missing.end());
    auto victims = policy->choose(layer, resident, needed, missing.size());
    if (victims.size() != missing.size()) throw std::runtime_error("native expert cache has insufficient slots");
    std::vector<std::pair<Id, Id>> loads;
    for (size_t i = 0; i < missing.size(); ++i) loads.emplace_back(original[victims[i]], missing[i]);
    return {std::move(loads), std::move(victims), std::move(missing)};
  }
  void commit(const Plan& work) {
    for (size_t i = 0; i < work.missing.size(); ++i) {
      mapping.write(work.victims[i], -1);
      mapping.write(work.missing[i], work.loads[i].first);
    }
  }
};

inline void bind_native_update(py::module_& m) {
  py::class_<NativeRanges, std::shared_ptr<NativeRanges>>(m, "NativeRanges")
      .def(py::init<>())
      .def("start", &NativeRanges::start)
      .def("status", [](NativeRanges& p) {
        const auto s = p.status();
        py::dict out; out["active"] = s.active; out["count"] = s.count;
        out["marker_errors"] = s.errors; out["inflight"] = s.inflight;
        out["marker_attempts"] = s.marker_attempts; return out;
      })
      .def("stop", [](NativeRanges& p) {
        std::pair<uint64_t, uint64_t> result;
        { py::gil_scoped_release release; result = p.stop(); }
        const auto [count, errors] = result;
        py::dict out; out["count"] = count; out["marker_errors"] = errors; return out;
      });
  py::class_<NativeTransfers, std::shared_ptr<NativeTransfers>>(m, "NativeTransfers")
      .def(py::init<NativeTransfers::Addresses, NativeTransfers::Addresses, std::vector<size_t>,
                    uintptr_t, const std::string&, bool>())
      .def("enqueue", [](NativeTransfers& t, const std::vector<std::pair<Id, Id>>& loads) {
        py::gil_scoped_release release;
        t.enqueue(loads);
      })
      .def("enqueue_all", &NativeTransfers::enqueue_all, py::call_guard<py::gil_scoped_release>())
      .def("set_profiler", [](NativeTransfers& t, std::shared_ptr<NativeRanges> p) { t.profiler = std::move(p); });
  m.def("native_update_task", [](std::shared_ptr<Policy> policy, std::shared_ptr<NativeTransfers> table,
                                 Id layer, py::array ids, py::object scores, py::array mapping,
                                 bool prefetch, Id cap, py::object owners) {
    auto* update = new NativeUpdate(policy, table, layer, ids, scores, mapping, prefetch, cap, owners);
    return py::capsule(&update->task, MoeNativeTask::capsule_name, [](void* ptr) {
      delete static_cast<NativeUpdate*>(static_cast<MoeNativeTask*>(ptr)->context);
    });
  });
  m.def("run_native_update_for_test", [](py::capsule capsule) {
    if (!capsule.name() || std::string(capsule.name()) != MoeNativeTask::capsule_name)
      throw std::invalid_argument("invalid native capsule");
    auto* task = static_cast<MoeNativeTask*>(capsule.get_pointer());
    py::gil_scoped_release release;
    task->run(task->context);
  });
}
