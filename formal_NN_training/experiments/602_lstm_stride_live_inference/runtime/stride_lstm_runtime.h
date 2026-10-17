#ifndef STRIDE_LSTM_RUNTIME_H
#define STRIDE_LSTM_RUNTIME_H

#include "stride_lstm_model_loader.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace stride_lstm {

struct RecurrentState {
  std::vector<float> hidden;
  std::vector<float> cell;
};

struct InferenceResult {
  std::uint64_t pc = 0;
  std::uint64_t line = 0;
  int emit = 0;
  std::uint64_t k = 0;
  std::vector<std::int64_t> deltas;
  std::vector<std::uint64_t> addresses;
  std::vector<float> hidden;
  std::vector<float> cell;
  std::uint64_t nanoseconds = 0;
};

struct RuntimeStats {
  std::uint64_t calls = 0;
  std::uint64_t generated_addresses = 0;
  std::uint64_t unique_pcs = 0;
  std::uint64_t peak_pc_states = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t bytes_per_pc_state = 0;
  std::uint64_t peak_recurrent_state_bytes = 0;
  std::uint64_t total_deployment_bytes = 0;
  double total_nanoseconds = 0.0;
  double mean_nanoseconds = 0.0;
  double p50_nanoseconds = 0.0;
  double p95_nanoseconds = 0.0;
  double p99_nanoseconds = 0.0;
  double maximum_nanoseconds = 0.0;
  double events_per_second = 0.0;
};

class StrideLSTMRuntime {
 public:
  explicit StrideLSTMRuntime(const std::string& model_path);

  InferenceResult Infer(std::uint64_t pc, std::uint64_t cache_line);
  void Reset();
  RuntimeStats stats() const;
  const FrozenModel& model() const { return model_; }

 private:
  FrozenModel model_;
  std::unordered_map<std::uint64_t, RecurrentState> states_;
  std::vector<std::uint64_t> timings_ns_;
  std::uint64_t calls_ = 0;
  std::uint64_t generated_addresses_ = 0;
};

std::string StaticOperationCountsJson(std::uint32_t hidden_size);

}  // namespace stride_lstm

#endif
