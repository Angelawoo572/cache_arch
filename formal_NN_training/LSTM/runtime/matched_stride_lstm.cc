#include "matched_stride_lstm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "champsim.h"

using namespace std;

namespace {
const char* MODEL_FORMAT = "matched_stride_lstm_runtime_v2";
}

const uint32_t MatchedStrideLSTM::INPUT_SIZE;
const uint32_t MatchedStrideLSTM::HIDDEN_SIZE;
const uint32_t MatchedStrideLSTM::CANDIDATE_SIZE;
const uint32_t MatchedStrideLSTM::TRACKER_CAPACITY;

MatchedStrideLSTM::MatchedStrideLSTM(string type)
    : Prefetcher(type), threshold_(1.0f), ready_(false), utility_bias_(0.0f),
      callbacks_(0), candidates_scored_(0), emitted_(0), tracker_evictions_(0)
{
    hidden_.fill(0.0f);
    cell_.fill(0.0f);
    load_model();
}

MatchedStrideLSTM::~MatchedStrideLSTM()
{
}

float MatchedStrideLSTM::sigmoid(float value)
{
    if (value >= 0.0f)
        return 1.0f / (1.0f + std::exp(-value));
    const float exponent = std::exp(value);
    return exponent / (1.0f + exponent);
}

float MatchedStrideLSTM::clip_unit(int64_t value, int64_t bound)
{
    value = std::max(-bound, std::min(bound, value));
    return static_cast<float>(value) / static_cast<float>(bound);
}

float MatchedStrideLSTM::pc_unit(uint64_t pc)
{
    const uint64_t mixed = pc ^ (pc >> 12) ^ (pc >> 24);
    return static_cast<float>(mixed & 4095ull) / 4095.0f;
}

void MatchedStrideLSTM::require_vector(istream& input, const char* expected,
                                       vector<float>& output, size_t count)
{
    string label;
    if (!(input >> label) || label != expected)
        throw runtime_error(string("expected model field ") + expected);
    output.resize(count);
    for (size_t index = 0; index < count; ++index) {
        if (!(input >> output[index]) || !std::isfinite(output[index]))
            throw runtime_error(string("invalid value in model field ") + expected);
    }
}

void MatchedStrideLSTM::load_model()
{
    const char* path = std::getenv("MATCHED_LSTM_MODEL_PATH");
    if (!path || !path[0])
        throw runtime_error("MATCHED_LSTM_MODEL_PATH is required");
    model_path_ = path;
    ifstream input(model_path_);
    if (!input)
        throw runtime_error("cannot open matched LSTM model: " + model_path_);

    string label, format;
    uint32_t input_size = 0, hidden_size = 0, candidate_size = 0;
    if (!(input >> label >> format) || label != "format" || format != MODEL_FORMAT)
        throw runtime_error("unsupported matched LSTM model format");
    if (!(input >> label >> input_size) || label != "input_size" || input_size != INPUT_SIZE)
        throw runtime_error("matched LSTM input-size mismatch");
    if (!(input >> label >> hidden_size) || label != "hidden_size" || hidden_size != HIDDEN_SIZE)
        throw runtime_error("matched LSTM hidden-size mismatch");
    if (!(input >> label >> candidate_size) || label != "candidate_size" || candidate_size != CANDIDATE_SIZE)
        throw runtime_error("matched LSTM candidate-size mismatch");
    if (!(input >> label >> threshold_) || label != "threshold" ||
        !std::isfinite(threshold_) || threshold_ < 0.0f || threshold_ > 1.0f)
        throw runtime_error("invalid matched LSTM threshold");

    require_vector(input, "weight_ih", weight_ih_, 4 * HIDDEN_SIZE * INPUT_SIZE);
    require_vector(input, "weight_hh", weight_hh_, 4 * HIDDEN_SIZE * HIDDEN_SIZE);
    require_vector(input, "bias_ih", bias_ih_, 4 * HIDDEN_SIZE);
    require_vector(input, "bias_hh", bias_hh_, 4 * HIDDEN_SIZE);
    require_vector(input, "projection_weight", projection_weight_,
                   HIDDEN_SIZE * (HIDDEN_SIZE + CANDIDATE_SIZE));
    require_vector(input, "projection_bias", projection_bias_, HIDDEN_SIZE);
    require_vector(input, "utility_weight", utility_weight_, HIDDEN_SIZE);
    if (!(input >> label >> utility_bias_) || label != "utility_bias" ||
        !std::isfinite(utility_bias_))
        throw runtime_error("invalid utility bias");
    string trailing;
    if (input >> trailing)
        throw runtime_error("unexpected trailing matched LSTM model content");
    ready_ = true;
    cerr << "[matched_stride_lstm] loaded live model " << model_path_
         << " (545 parameters; one stride candidate; threshold="
         << threshold_ << ")" << endl;
}

void MatchedStrideLSTM::update_lstm(const array<float, INPUT_SIZE>& input)
{
    array<float, 4 * HIDDEN_SIZE> gates;
    for (uint32_t row = 0; row < 4 * HIDDEN_SIZE; ++row) {
        float value = bias_ih_[row] + bias_hh_[row];
        for (uint32_t column = 0; column < INPUT_SIZE; ++column)
            value += weight_ih_[row * INPUT_SIZE + column] * input[column];
        for (uint32_t column = 0; column < HIDDEN_SIZE; ++column)
            value += weight_hh_[row * HIDDEN_SIZE + column] * hidden_[column];
        gates[row] = value;
    }

    array<float, HIDDEN_SIZE> next_hidden;
    array<float, HIDDEN_SIZE> next_cell;
    for (uint32_t index = 0; index < HIDDEN_SIZE; ++index) {
        const float input_gate = sigmoid(gates[index]);
        const float forget_gate = sigmoid(gates[HIDDEN_SIZE + index]);
        const float cell_gate = std::tanh(gates[2 * HIDDEN_SIZE + index]);
        const float output_gate = sigmoid(gates[3 * HIDDEN_SIZE + index]);
        next_cell[index] = forget_gate * cell_[index] + input_gate * cell_gate;
        next_hidden[index] = output_gate * std::tanh(next_cell[index]);
    }
    hidden_ = next_hidden;
    cell_ = next_cell;
}

float MatchedStrideLSTM::score_candidate(int64_t delta, float repeat) const
{
    array<float, HIDDEN_SIZE + CANDIDATE_SIZE> joined;
    for (uint32_t index = 0; index < HIDDEN_SIZE; ++index)
        joined[index] = hidden_[index];
    joined[HIDDEN_SIZE] = clip_unit(delta, 64);
    joined[HIDDEN_SIZE + 1] = repeat;

    array<float, HIDDEN_SIZE> projected;
    for (uint32_t row = 0; row < HIDDEN_SIZE; ++row) {
        float value = projection_bias_[row];
        for (uint32_t column = 0; column < joined.size(); ++column)
            value += projection_weight_[row * joined.size() + column] * joined[column];
        projected[row] = std::tanh(value);
    }
    float utility = utility_bias_;
    for (uint32_t index = 0; index < HIDDEN_SIZE; ++index)
        utility += utility_weight_[index] * projected[index];
    return sigmoid(utility);
}

void MatchedStrideLSTM::touch_tracker(uint64_t pc, uint64_t line, int64_t stride)
{
    auto existing = trackers_.find(pc);
    if (existing != trackers_.end()) {
        tracker_lru_.erase(existing->second.lru_position);
        tracker_lru_.push_front(pc);
        existing->second.last_line = line;
        existing->second.last_stride = stride;
        existing->second.lru_position = tracker_lru_.begin();
        return;
    }
    if (trackers_.size() >= TRACKER_CAPACITY) {
        const uint64_t victim = tracker_lru_.back();
        tracker_lru_.pop_back();
        trackers_.erase(victim);
        ++tracker_evictions_;
    }
    tracker_lru_.push_front(pc);
    trackers_[pc] = Tracker{line, stride, tracker_lru_.begin()};
}

void MatchedStrideLSTM::invoke_prefetcher(uint64_t pc, uint64_t address,
                                          uint8_t /*cache_hit*/, uint8_t /*type*/,
                                          vector<uint64_t>& pref_addr)
{
    if (!ready_)
        return;
    ++callbacks_;
    const uint64_t line = address >> LOG2_BLOCK_SIZE;
    const uint64_t page = address >> LOG2_PAGE_SIZE;
    int64_t stride = 0;
    int64_t previous_stride = 0;
    const auto tracker = trackers_.find(pc);
    if (tracker != trackers_.end()) {
        if (line >= tracker->second.last_line)
            stride = static_cast<int64_t>(line - tracker->second.last_line);
        else
            stride = -static_cast<int64_t>(tracker->second.last_line - line);
        previous_stride = tracker->second.last_stride;
    }

    const array<float, INPUT_SIZE> features = {{
        pc_unit(pc),
        clip_unit(stride, 256),
        clip_unit(previous_stride, 256),
        static_cast<float>(line & 63ull) / 63.0f,
    }};
    update_lstm(features);

    if (stride != 0) {
        const int64_t target_line = static_cast<int64_t>(line) + stride;
        if (target_line > 0 && (static_cast<uint64_t>(target_line) >> 6) == page) {
            const float repeat = (stride == previous_stride) ? 1.0f : 0.5f;
            const float score = score_candidate(stride, repeat);
            ++candidates_scored_;
            if (score >= threshold_) {
                pref_addr.push_back(static_cast<uint64_t>(target_line) << LOG2_BLOCK_SIZE);
                ++emitted_;
            }
        }
    }
    // Match the audited stride tracker's zero-stride update/LRU semantics.
    if (tracker == trackers_.end() || stride != 0)
        touch_tracker(pc, line, stride);
}

void MatchedStrideLSTM::dump_stats()
{
    cout << "matched_stride_lstm_callbacks " << callbacks_ << endl
         << "matched_stride_lstm_candidates_scored " << candidates_scored_ << endl
         << "matched_stride_lstm_emitted " << emitted_ << endl
         << "matched_stride_lstm_tracker_evictions " << tracker_evictions_ << endl;
}

void MatchedStrideLSTM::print_config()
{
    cout << "matched_stride_lstm_model " << model_path_ << endl
         << "matched_stride_lstm_parameter_count 545" << endl
         << "matched_stride_lstm_tracker_capacity " << TRACKER_CAPACITY << endl
         << "matched_stride_lstm_candidate_source current_per_pc_stride_only" << endl
         << "matched_stride_lstm_threshold " << threshold_ << endl
         << "matched_stride_lstm_max_degree 1" << endl
         << "matched_stride_lstm_runtime_inputs pc,address,causal_pc_address_history" << endl;
}
