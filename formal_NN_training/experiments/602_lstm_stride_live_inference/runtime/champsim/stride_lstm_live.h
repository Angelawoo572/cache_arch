#ifndef STRIDE_LSTM_LIVE_H
#define STRIDE_LSTM_LIVE_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "prefetcher.h"
#include "stride_lstm_runtime.h"

class CACHE;

class StrideLSTMLive : public Prefetcher
{
public:
    StrideLSTMLive(std::string type, CACHE* cache);
    ~StrideLSTMLive();

    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit,
                           uint8_t type, std::vector<uint64_t>& pref_addr);
    void dump_stats();
    void print_config();

private:
    CACHE* cache_;
    std::unique_ptr<stride_lstm::StrideLSTMRuntime> runtime_;
    std::string model_path_;
    std::string state_mode_;
    bool measurement_started_ = false;
    bool stats_dumped_ = false;
    uint64_t warmup_callbacks_ = 0;
    uint64_t measured_callbacks_ = 0;
};

#endif

