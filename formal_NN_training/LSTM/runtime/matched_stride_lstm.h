#ifndef MATCHED_STRIDE_LSTM_H
#define MATCHED_STRIDE_LSTM_H

#include <array>
#include <cstdint>
#include <istream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

#include "prefetcher.h"

// Live, one-trace LSTM used only by the controlled 602 matched-input study.
// Runtime inputs are restricted to the PC/address callback stream also seen by
// the repository's stride prefetcher.  Model weights and the bounded PC/delta
// table are learned from the earlier training window and loaded from a text
// artifact named by MATCHED_LSTM_MODEL_PATH.  It scores only the current
// per-PC stride candidate; there is no learned PC/delta side table.
class MatchedStrideLSTM : public Prefetcher
{
public:
    explicit MatchedStrideLSTM(std::string type);
    ~MatchedStrideLSTM();

    void invoke_prefetcher(uint64_t pc, uint64_t address, uint8_t cache_hit,
                           uint8_t type, std::vector<uint64_t>& pref_addr);
    void dump_stats();
    void print_config();

private:
    static const uint32_t INPUT_SIZE = 4;
    static const uint32_t HIDDEN_SIZE = 8;
    static const uint32_t CANDIDATE_SIZE = 2;
    static const uint32_t TRACKER_CAPACITY = 64;

    struct Tracker {
        uint64_t last_line;
        int64_t last_stride;
        std::list<uint64_t>::iterator lru_position;
    };

    void load_model();
    void require_vector(std::istream& input, const char* expected,
                        std::vector<float>& output, size_t count);
    void update_lstm(const std::array<float, INPUT_SIZE>& input);
    float score_candidate(int64_t delta, float repeat) const;
    void touch_tracker(uint64_t pc, uint64_t line, int64_t stride);
    static float sigmoid(float value);
    static float clip_unit(int64_t value, int64_t bound);
    static float pc_unit(uint64_t pc);

    std::string model_path_;
    float threshold_;
    bool ready_;

    std::vector<float> weight_ih_;
    std::vector<float> weight_hh_;
    std::vector<float> bias_ih_;
    std::vector<float> bias_hh_;
    std::vector<float> projection_weight_;
    std::vector<float> projection_bias_;
    std::vector<float> utility_weight_;
    float utility_bias_;
    std::array<float, HIDDEN_SIZE> hidden_;
    std::array<float, HIDDEN_SIZE> cell_;

    std::unordered_map<uint64_t, Tracker> trackers_;
    std::list<uint64_t> tracker_lru_;
    uint64_t callbacks_;
    uint64_t candidates_scored_;
    uint64_t emitted_;
    uint64_t tracker_evictions_;
};

#endif
