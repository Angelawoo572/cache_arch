#include "stride_lstm_runtime.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace stride_lstm {
namespace {

const std::uint64_t kLineMask = (1ULL << 58U) - 1ULL;

float Sigmoid(float value) {
  if (value >= 0.0F) {
    const float inverse = std::exp(-value);
    return 1.0F / (1.0F + inverse);
  }
  const float exponential = std::exp(value);
  return exponential / (1.0F + exponential);
}

float DotRow(const Tensor& matrix, std::size_t row,
             const std::vector<float>& vector) {
  const std::size_t width = matrix.shape.at(1);
  float total = 0.0F;
  const std::size_t offset = row * width;
  for (std::size_t column = 0; column < width; ++column) {
    total += matrix.data[offset + column] * vector[column];
  }
  return total;
}

std::vector<float> Linear(const Tensor& weight, const Tensor& bias,
                          const std::vector<float>& input) {
  std::vector<float> output(weight.shape.at(0), 0.0F);
  for (std::size_t row = 0; row < output.size(); ++row) {
    output[row] = bias.data[row] + DotRow(weight, row, input);
  }
  return output;
}

std::int64_t CoordinateToDelta(float coordinate) {
  if (!std::isfinite(coordinate)) {
    throw std::runtime_error("neural delta coordinate is not finite");
  }
  const double magnitude = std::expm1(std::fabs(static_cast<double>(coordinate)));
  if (!std::isfinite(magnitude) ||
      magnitude > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::runtime_error("neural delta exceeds address domain");
  }
  const std::int64_t rounded =
      static_cast<std::int64_t>(std::nearbyint(magnitude));
  return coordinate < 0.0F ? -rounded : rounded;
}

std::uint64_t ApplySignedDelta(std::uint64_t line, std::int64_t delta) {
  return (line + static_cast<std::uint64_t>(delta)) & kLineMask;
}

double Percentile(const std::vector<std::uint64_t>& sorted, double fraction) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double position = fraction * static_cast<double>(sorted.size() - 1U);
  const std::size_t low = static_cast<std::size_t>(std::floor(position));
  const std::size_t high = static_cast<std::size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(low);
  return static_cast<double>(sorted[low]) * (1.0 - weight) +
         static_cast<double>(sorted[high]) * weight;
}

}  // namespace

StrideLSTMRuntime::StrideLSTMRuntime(const std::string& model_path) {
  model_.Load(model_path);
}

void StrideLSTMRuntime::Reset() {
  states_.clear();
}

InferenceResult StrideLSTMRuntime::Infer(std::uint64_t pc,
                                         std::uint64_t cache_line) {
  const auto start = std::chrono::steady_clock::now();
  const std::size_t h = model_.hidden_size();
  const std::size_t f = model_.feature_width();
  std::vector<float> features(f, 0.0F);
  const std::uint64_t address = cache_line << 6U;
  for (std::size_t bit = 0; bit < 64U; ++bit) {
    features[bit] = static_cast<float>((pc >> bit) & 1ULL);
    features[64U + bit] =
        static_cast<float>((address >> bit) & 1ULL);
  }
  std::vector<float> projected = Linear(
      model_.tensor("input_projection.weight"),
      model_.tensor("input_projection.bias"), features);
  for (float& value : projected) {
    value = std::tanh(value);
  }
  RecurrentState previous;
  const auto found = states_.find(pc);
  if (found == states_.end()) {
    previous.hidden.assign(h, 0.0F);
    previous.cell.assign(h, 0.0F);
  } else {
    previous = found->second;
  }
  const Tensor& w_ih = model_.tensor("encoder_lstm.weight_ih_l0");
  const Tensor& w_hh = model_.tensor("encoder_lstm.weight_hh_l0");
  const Tensor& b_ih = model_.tensor("encoder_lstm.bias_ih_l0");
  const Tensor& b_hh = model_.tensor("encoder_lstm.bias_hh_l0");
  std::vector<float> gates(4U * h, 0.0F);
  for (std::size_t row = 0; row < gates.size(); ++row) {
    gates[row] = b_ih.data[row] + b_hh.data[row] +
                 DotRow(w_ih, row, projected) +
                 DotRow(w_hh, row, previous.hidden);
  }
  RecurrentState current;
  current.hidden.resize(h);
  current.cell.resize(h);
  for (std::size_t index = 0; index < h; ++index) {
    const float input_gate = Sigmoid(gates[index]);
    const float forget_gate = Sigmoid(gates[h + index]);
    const float candidate = std::tanh(gates[2U * h + index]);
    const float output_gate = Sigmoid(gates[3U * h + index]);
    current.cell[index] =
        forget_gate * previous.cell[index] + input_gate * candidate;
    current.hidden[index] =
        output_gate * std::tanh(current.cell[index]);
  }
  states_[pc] = current;

  const std::vector<float> emit_logits = Linear(
      model_.tensor("emit_head.weight"),
      model_.tensor("emit_head.bias"), current.hidden);
  const int emit = emit_logits[1] > emit_logits[0] ? 1 : 0;
  std::uint64_t count = 0;
  if (emit != 0) {
    const std::vector<float> log_mean = Linear(
        model_.tensor("log_count_mean.weight"),
        model_.tensor("log_count_mean.bias"), current.hidden);
    const double raw_count = std::exp(static_cast<double>(log_mean[0]));
    if (!std::isfinite(raw_count) ||
        raw_count > static_cast<double>(
            std::numeric_limits<std::uint64_t>::max())) {
      throw std::runtime_error("positive-count prediction exceeds host domain");
    }
    count = static_cast<std::uint64_t>(std::nearbyint(raw_count));
    if (count < 1U) {
      count = 1U;
    }
  }

  InferenceResult result;
  result.pc = pc;
  result.line = cache_line;
  result.emit = emit;
  result.k = count;
  result.hidden = current.hidden;
  result.cell = current.cell;
  std::vector<float> decoder_state = current.hidden;
  const Tensor& gru_w_ih =
      model_.tensor("action_decoder.action_cell.weight_ih");
  const Tensor& gru_w_hh =
      model_.tensor("action_decoder.action_cell.weight_hh");
  const Tensor& gru_b_ih =
      model_.tensor("action_decoder.action_cell.bias_ih");
  const Tensor& gru_b_hh =
      model_.tensor("action_decoder.action_cell.bias_hh");
  for (std::uint64_t slot = 0; slot < count; ++slot) {
    const std::vector<float> coordinate_value = Linear(
        model_.tensor("action_decoder.delta_head.weight"),
        model_.tensor("action_decoder.delta_head.bias"), decoder_state);
    const float coordinate = coordinate_value[0];
    const std::int64_t delta = CoordinateToDelta(coordinate);
    result.deltas.push_back(delta);
    result.addresses.push_back(ApplySignedDelta(cache_line, delta) << 6U);

    std::vector<float> next(h, 0.0F);
    for (std::size_t index = 0; index < h; ++index) {
      const float input_r =
          gru_b_ih.data[index] + gru_w_ih.data[index] * coordinate;
      const float input_z =
          gru_b_ih.data[h + index] +
          gru_w_ih.data[h + index] * coordinate;
      const float input_n =
          gru_b_ih.data[2U * h + index] +
          gru_w_ih.data[2U * h + index] * coordinate;
      const float hidden_r =
          gru_b_hh.data[index] + DotRow(gru_w_hh, index, decoder_state);
      const float hidden_z =
          gru_b_hh.data[h + index] +
          DotRow(gru_w_hh, h + index, decoder_state);
      const float hidden_n =
          gru_b_hh.data[2U * h + index] +
          DotRow(gru_w_hh, 2U * h + index, decoder_state);
      const float reset = Sigmoid(input_r + hidden_r);
      const float update = Sigmoid(input_z + hidden_z);
      const float candidate = std::tanh(input_n + reset * hidden_n);
      next[index] =
          (1.0F - update) * candidate + update * decoder_state[index];
    }
    decoder_state.swap(next);
  }
  const auto stop = std::chrono::steady_clock::now();
  result.nanoseconds = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          stop - start).count());
  ++calls_;
  generated_addresses_ += result.addresses.size();
  timings_ns_.push_back(result.nanoseconds);
  return result;
}

RuntimeStats StrideLSTMRuntime::stats() const {
  RuntimeStats result;
  result.calls = calls_;
  result.generated_addresses = generated_addresses_;
  result.unique_pcs = states_.size();
  result.peak_pc_states = states_.size();
  result.weight_bytes = model_.weight_bytes();
  result.bytes_per_pc_state = 2ULL * model_.hidden_size() * sizeof(float);
  result.peak_recurrent_state_bytes =
      result.peak_pc_states * result.bytes_per_pc_state;
  result.total_deployment_bytes =
      result.weight_bytes + result.peak_recurrent_state_bytes;
  std::vector<std::uint64_t> sorted = timings_ns_;
  std::sort(sorted.begin(), sorted.end());
  for (std::uint64_t value : sorted) {
    result.total_nanoseconds += static_cast<double>(value);
  }
  result.mean_nanoseconds =
      result.calls ? result.total_nanoseconds / result.calls : 0.0;
  result.p50_nanoseconds = Percentile(sorted, 0.50);
  result.p95_nanoseconds = Percentile(sorted, 0.95);
  result.p99_nanoseconds = Percentile(sorted, 0.99);
  result.maximum_nanoseconds =
      sorted.empty() ? 0.0 : static_cast<double>(sorted.back());
  result.events_per_second =
      result.total_nanoseconds > 0.0
          ? static_cast<double>(result.calls) * 1.0e9 /
                result.total_nanoseconds
          : 0.0;
  return result;
}

std::string StaticOperationCountsJson(std::uint32_t hidden_size) {
  const std::uint64_t h = hidden_size;
  const std::uint64_t fixed_multiplies =
      h * 128ULL + 8ULL * h * h + 3ULL * h;
  const std::uint64_t decoder_multiplies_per_slot =
      3ULL * h * h + 4ULL * h;
  std::ostringstream output;
  output << "{\"fixed_multiplies_per_callback\":" << fixed_multiplies
         << ",\"fixed_adds_per_callback\":" << fixed_multiplies
         << ",\"fixed_sigmoid_activations\":" << 3ULL * h
         << ",\"fixed_tanh_activations\":" << 3ULL * h
         << ",\"decoder_multiplies_per_slot\":"
         << decoder_multiplies_per_slot
         << ",\"decoder_adds_per_slot\":" << decoder_multiplies_per_slot
         << ",\"decoder_sigmoid_activations_per_slot\":" << 2ULL * h
         << ",\"decoder_tanh_activations_per_slot\":" << h
         << ",\"decoder_expm1_per_slot\":1}";
  return output.str();
}

}  // namespace stride_lstm

