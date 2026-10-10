// SPDX-License-Identifier: Apache-2.0
// CPU-only LRC state. Keep GIL held: callbacks must not concurrently mutate it.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace py = pybind11;
using Id = int64_t;
using Counts = std::vector<std::pair<Id, Id>>;

struct Layer {
  std::vector<Id> freq, last;
  std::vector<float> ema;
  std::vector<double> router;
  std::deque<Counts> queue;
  Id step = 0;
  explicit Layer(Id n) : freq(n, 0), last(n, -1), ema(n, 0), router(n, 0) {}
};

class Policy {
  Id experts_, window_;
  double beta_, recent_, ema_weight_, router_weight_, age_weight_;
  bool scalar_f32_;
  std::vector<Layer> layers_;

  Layer& layer(Id i) {
    if (i < 0) i += static_cast<Id>(layers_.size());
    if (i < 0 || i >= static_cast<Id>(layers_.size())) throw py::index_error("layer index out of range");
    return layers_[i];
  }
  Id expert(Id i) const {
    if (i < 0) i += experts_;
    if (i < 0 || i >= experts_) throw py::index_error("expert index out of range");
    return i;
  }
  void trim(Layer& s) {
    while (static_cast<Id>(s.queue.size()) > window_) {
      for (auto [e, count] : s.queue.front()) s.freq[e] -= count;
      s.queue.pop_front();
    }
  }
  double score(const Layer& s, Id e) const {
    Id age = s.last[e] < 0 ? 0 : s.step - s.last[e];
    if (scalar_f32_) {
      // NumPy 2 weak Python-scalar promotion, matching hotness().
      float value = static_cast<float>(recent_ * s.freq[e]) + static_cast<float>(ema_weight_) * s.ema[e];
      value = value + static_cast<float>(router_weight_ * s.router[e]);
      value = value - static_cast<float>(age_weight_ * age);
      return value;
    }
    return recent_ * s.freq[e] + ema_weight_ * static_cast<double>(s.ema[e]) + router_weight_ * s.router[e] -
           age_weight_ * age;
  }

 public:
  Policy(Id layers, Id experts, Id window, double beta, double recent, double ema_weight, double router_weight,
         double age_weight, bool scalar_f32)
      : experts_(experts),
        window_(window),
        beta_(beta),
        recent_(recent),
        ema_weight_(ema_weight),
        router_weight_(router_weight),
        age_weight_(age_weight),
        scalar_f32_(scalar_f32) {
    if (layers < 1 || experts < 1 || window < 1 || !(beta >= 0 && beta < 1))
      throw py::value_error("Invalid LRC dimensions or EMA beta");
    for (Id i = 0; i < layers; ++i) layers_.emplace_back(experts);
  }
  Id add_layer() {
    layers_.emplace_back(experts_);
    return static_cast<Id>(layers_.size()) - 1;
  }
  Id num_layers() const { return layers_.size(); }
  Id step(Id idx) { return layer(idx).step; }

  std::unordered_set<Id> observe(Id idx, const std::vector<std::vector<Id>>& rows, const py::object& score_obj) {
    auto& s = layer(idx);
    bool has_scores = !score_obj.is_none();
    std::vector<std::vector<double>> scores;
    if (has_scores) scores = score_obj.cast<std::vector<std::vector<double>>>();
    std::unordered_set<Id> unique;
    for (size_t r = 0; r < rows.size(); ++r) {
      const auto& row = rows[r];
      ++s.step;
      for (auto& v : s.ema) v = v * static_cast<float>(beta_);
      Counts counts;
      for (Id e : row) {
        if (e < 0 || e >= experts_) continue;
        unique.insert(e);
        auto found = std::find_if(counts.begin(), counts.end(), [e](const auto& v) { return v.first == e; });
        if (found == counts.end()) {
          counts.emplace_back(e, 1);
          s.ema[e] = s.ema[e] + static_cast<float>(1.0 - beta_);
        } else {
          ++found->second;
        }
        ++s.freq[e];
        s.last[e] = s.step;
      }
      s.queue.push_back(counts);
      // Match the reference's error ordering (state is already updated).
      if (has_scores) {
        if (r >= scores.size()) throw py::index_error("router score row missing");
        if (scores[r].size() != row.size()) throw py::value_error("router score width does not match topk ID width");
        for (size_t c = 0; c < row.size(); ++c) {
          Id e = row[c];
          if (e >= 0 && e < experts_) s.router[e] = scores[r][c];
        }
      }
      trim(s);
    }
    return unique;
  }

  std::unordered_set<Id> observe_global(Id idx, const std::vector<Id>& counts, const py::object& score_obj) {
    if (static_cast<Id>(counts.size()) != experts_) throw py::value_error("global_counts has wrong shape");
    for (auto v : counts)
      if (v < 0) throw py::value_error("global_counts must be non-negative");
    bool has_scores = !score_obj.is_none();
    std::vector<float> scores;
    if (has_scores) {
      scores = score_obj.cast<std::vector<float>>();
      if (scores.size() != counts.size()) throw py::value_error("global_router_score has wrong shape");
    }
    auto& s = layer(idx);
    ++s.step;
    Counts snapshot;
    std::unordered_set<Id> active;
    for (Id e = 0; e < experts_; ++e) {
      s.ema[e] = s.ema[e] * static_cast<float>(beta_);
      float increment = static_cast<float>(counts[e]) * static_cast<float>(1.0 - beta_);
      s.ema[e] = s.ema[e] + increment;
      if (counts[e]) {
        active.insert(e);
        snapshot.emplace_back(e, counts[e]);
        s.freq[e] += counts[e];
        s.last[e] = s.step;
        if (has_scores) s.router[e] = scores[e];
      }
    }
    s.queue.push_back(snapshot);
    trim(s);
    return active;
  }

  std::vector<Id> choose(Id idx, const std::vector<Id>& residents, const std::unordered_set<Id>& protected_ids,
                         Id count) {
    if (count < 0) throw py::value_error("victim count must be >= 0");
    if (!count) return {};
    const auto& s = layer(idx);
    std::vector<std::tuple<double, Id, Id>> keys;
    for (Id e : residents) {
      if (!protected_ids.count(e)) keys.emplace_back(score(s, expert(e)), s.last[expert(e)], e);
    }
    std::stable_sort(keys.begin(), keys.end());
    std::vector<Id> result;
    for (Id i = 0; i < std::min(count, static_cast<Id>(keys.size())); ++i) result.push_back(std::get<2>(keys[i]));
    return result;
  }
  double hotness(Id idx, Id e) { return score(layer(idx), expert(e)); }
  std::vector<float> hotness_array(Id idx) {
    const auto& s = layer(idx);
    std::vector<float> result(experts_);
    for (Id e = 0; e < experts_; ++e) {
      float last = static_cast<float>(s.last[e]);
      float age = last < 0 ? 0 : static_cast<float>(s.step) - last;
      float value =
          static_cast<float>(recent_) * static_cast<float>(s.freq[e]) + static_cast<float>(ema_weight_) * s.ema[e];
      value = value + static_cast<float>(router_weight_) * static_cast<float>(s.router[e]);
      result[e] = value - static_cast<float>(age_weight_) * age;
    }
    return result;
  }
  void seed(Id idx, Id eid, Id freq, double weight) {
    auto& s = layer(idx);
    Id e = expert(eid);
    s.freq[e] = freq;
    s.router[e] = weight;
    s.last[e] = s.step;
  }
  py::dict snapshot(Id idx) {
    const auto& s = layer(idx);
    py::dict d;
    d["freq"] = s.freq;
    d["ema"] = s.ema;
    d["router_score"] = s.router;
    d["last_used"] = s.last;
    d["step"] = s.step;
    d["recent_queue"] = s.queue;
    return d;
  }
};

PYBIND11_MODULE(_vllm_ascend_lrc, m) {
  m.attr("api_version") = 1;
  py::class_<Policy>(m, "Policy")
      .def(py::init<Id, Id, Id, double, double, double, double, double, bool>())
      .def("add_layer", &Policy::add_layer)
      .def("num_layers", &Policy::num_layers)
      .def("step", &Policy::step)
      .def("observe", &Policy::observe)
      .def("observe_global", &Policy::observe_global)
      .def("choose", &Policy::choose)
      .def("hotness", &Policy::hotness)
      .def("hotness_array", &Policy::hotness_array)
      .def("seed", &Policy::seed)
      .def("snapshot", &Policy::snapshot);
}
