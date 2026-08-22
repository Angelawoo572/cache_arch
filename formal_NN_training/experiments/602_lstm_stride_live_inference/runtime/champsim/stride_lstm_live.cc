#include "stride_lstm_live.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "cache.h"
#include "champsim.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

StrideLSTMLive::StrideLSTMLive(string type, CACHE* cache)
    : Prefetcher(type), cache_(cache)
{
    const char* model = std::getenv("STRIDE_LSTM_MODEL_BIN");
    if (model == nullptr || model[0] == '\0')
        throw std::runtime_error("STRIDE_LSTM_MODEL_BIN is required");
    model_path_ = model;
    const char* mode = std::getenv("STRIDE_LSTM_STATE_MODE");
    state_mode_ = (mode && mode[0]) ? mode : "parity";
    if (state_mode_ != "parity" && state_mode_ != "realistic_state_warmup")
        throw std::runtime_error(
            "STRIDE_LSTM_STATE_MODE must be parity or realistic_state_warmup");
    runtime_.reset(new stride_lstm::StrideLSTMRuntime(model_path_));
    print_config();
}

StrideLSTMLive::~StrideLSTMLive() = default;

void StrideLSTMLive::invoke_prefetcher(
    uint64_t pc, uint64_t address, uint8_t /*cache_hit*/, uint8_t access_type,
    vector<uint64_t>& pref_addr)
{
    // CACHE also calls this interface for prefetch-queue accesses. Eligibility
    // is an interface guard; only PC and byte address enter the frozen model.
    if (access_type != LOAD)
        return;

    const bool measured = warmup_complete[cache_->cpu] != 0;
    if (!measured) {
        ++warmup_callbacks_;
        if (state_mode_ == "realistic_state_warmup")
            runtime_->Infer(pc, address >> LOG2_BLOCK_SIZE);
        return;
    }
    if (!measurement_started_) {
        if (state_mode_ == "parity")
            runtime_->Reset();
        measurement_started_ = true;
        cout << "stride_lstm_live_measurement_start state_mode "
             << state_mode_ << endl;
    }

    const stride_lstm::InferenceResult result =
        runtime_->Infer(pc, address >> LOG2_BLOCK_SIZE);
    ++measured_callbacks_;
    pref_addr.insert(
        pref_addr.end(), result.addresses.begin(), result.addresses.end());
}

void StrideLSTMLive::print_config()
{
    cout << "stride_lstm_live_model " << model_path_ << endl;
    cout << "stride_lstm_live_state_mode " << state_mode_ << endl;
    cout << "stride_lstm_live_weights frozen" << endl;
    cout << "stride_lstm_live_nn_inference_latency_cycles 0" << endl;
}

void StrideLSTMLive::dump_stats()
{
    if (stats_dumped_)
        return;
    stats_dumped_ = true;
    const stride_lstm::RuntimeStats stats = runtime_->stats();
    cout << "stride_lstm_live_warmup_callbacks " << warmup_callbacks_ << endl;
    cout << "stride_lstm_live_measured_callbacks " << measured_callbacks_ << endl;
    cout << "stride_lstm_live_inference_calls " << stats.calls << endl;
    cout << "stride_lstm_live_generated_addresses "
         << stats.generated_addresses << endl;
    cout << "stride_lstm_live_unique_pcs " << stats.unique_pcs << endl;
    cout << "stride_lstm_live_peak_pc_states " << stats.peak_pc_states << endl;
    cout << "stride_lstm_live_bytes_per_pc_state "
         << stats.bytes_per_pc_state << endl;
    cout << "stride_lstm_live_peak_recurrent_state_bytes "
         << stats.peak_recurrent_state_bytes << endl;
    cout << "stride_lstm_live_weight_bytes " << stats.weight_bytes << endl;
    cout << "stride_lstm_live_total_deployment_bytes "
         << stats.total_deployment_bytes << endl;
    cout << "stride_lstm_live_host_total_ns " << stats.total_nanoseconds << endl;
    cout << "stride_lstm_live_host_mean_ns " << stats.mean_nanoseconds << endl;
    cout << "stride_lstm_live_host_p50_ns " << stats.p50_nanoseconds << endl;
    cout << "stride_lstm_live_host_p95_ns " << stats.p95_nanoseconds << endl;
    cout << "stride_lstm_live_host_p99_ns " << stats.p99_nanoseconds << endl;
    cout << "stride_lstm_live_host_max_ns "
         << stats.maximum_nanoseconds << endl;
    cout << "stride_lstm_live_host_events_per_second "
         << stats.events_per_second << endl;
    cout << "stride_lstm_live_static_operations "
         << stride_lstm::StaticOperationCountsJson(
                runtime_->model().hidden_size()) << endl;
}
