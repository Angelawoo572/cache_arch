#ifndef LLC_PRESSURE_PREFETCH_H
#define LLC_PRESSURE_PREFETCH_H

#include "learned_llc.h"

namespace llc_learned {

using PressureFeatures = std::array<float, 4>;

// Called at LLC ingress, including for events that later miss the predictor FIFO.
// The candidate and scoring policies do not affect the observation order.
template <size_t Cores>
class PressureArrivals {
public:
    PressureFeatures observe(size_t core, uint64_t cycle, uint32_t mshr_occupancy)
    {
        auto& clock = local[core];
        const PressureFeatures features{{clock.gap(cycle, 8), clock.gap(cycle, 32), shared.gap(cycle, 8),
                                        static_cast<float>(std::min<uint32_t>(mshr_occupancy, 32)) / 32}};
        clock.push(cycle);
        shared.push(cycle);
        return features;
    }

private:
    template <size_t Size>
    class ArrivalClock {
    public:
        float gap(uint64_t cycle, size_t width) const
        {
            const auto count = std::min(width, size);
            if (!count)
                return 0;
            const auto prior = cycles[(next + Size - count) % Size];
            const auto average = cycle >= prior ? static_cast<float>(static_cast<double>(cycle - prior) / count) : 0;
            return std::min(1.0f, std::log2(1.0f + average) / 9);
        }

        void push(uint64_t cycle)
        {
            cycles[next] = cycle;
            next = (next + 1) % Size;
            size = std::min(size + 1, Size);
        }

    private:
        std::array<uint64_t, Size> cycles{};
        size_t next = 0;
        size_t size = 0;
    };

    std::array<ArrivalClock<32>, Cores> local;
    ArrivalClock<8> shared;
};

constexpr size_t pressure_max_hidden = 16;
constexpr size_t pressure_head_width = 8;

struct PressureState {
    std::array<float, pressure_max_hidden> hidden{};
    std::array<float, pressure_max_hidden> cell{};
    uint64_t events = 0;
};

class PressureModel {
public:
    PressureModel(const char* path, const std::string& expected_mode)
    {
        if (!path)
            throw std::runtime_error("pressure scorer requires a model file");
        std::ifstream input(path);
        std::string magic, mode;
        size_t event_width, head, source_width, pressure_width, cap, pc_count;
        if (!(input >> magic >> mode >> event_width >> hidden >> head >> source_width >> pressure_width
                >> cap >> reset_events >> pc_count)
            || magic != "LLC_LSTM_TIMING_V1" || mode != expected_mode || (mode != "event" && mode != "lstm")
            || event_width != event_size || (hidden != 8 && hidden != pressure_max_hidden)
            || head != pressure_head_width || source_width != candidate_size || pressure_width != 4
            || cap != max_candidates || pc_count > 16)
            throw std::runtime_error("invalid or incompatible pressure model header");
        recurrent = mode == "lstm";

        for (auto& coefficient : unknown)
            coefficient = weight(input);
        for (size_t i = 0; i < pc_count; ++i) {
            PcEmbedding entry{};
            if (!(input >> entry.tag) || (!embeddings.empty() && embeddings.back().tag >= entry.tag))
                throw std::runtime_error("invalid or unordered pressure model PC tag");
            for (auto& coefficient : entry.values)
                coefficient = weight(input);
            embeddings.push_back(entry);
        }
        load(input, weights_input, 4 * hidden * event_size);
        load(input, weights_hidden, 4 * hidden * hidden);
        load(input, cell_bias, 4 * hidden);
        load(input, head_event, pressure_head_width * context_size());
        load(input, head_bias, pressure_head_width);
        load(input, head_source, pressure_head_width * candidate_size);
        load(input, reduction, pressure_head_width);
        output_bias = weight(input);
        std::string extra;
        if (input >> extra)
            throw std::runtime_error("trailing tokens in pressure model");
    }

    size_t hidden_width() const { return hidden; }
    size_t pc_entries() const { return embeddings.size(); }

    std::array<int32_t, max_candidates> score(const Observation& event, const PressureFeatures& pressure,
                                             PressureState& state, bool use_hidden = true) const
    {
        auto current = event.features;
        const auto current_pc = lookup(event.pc);
        std::copy(current_pc.begin(), current_pc.end(), current.begin());
        if (recurrent)
            update(current, state);

        std::array<float, pressure_head_width> projection;
        for (size_t unit = 0; unit < projection.size(); ++unit) {
            const auto row = unit * context_size();
            float value = head_bias[unit];
            for (size_t j = 0; j < current.size(); ++j)
                value += head_event[row + j] * current[j];
            for (size_t j = 0; j < pressure.size(); ++j)
                value += head_event[row + event_size + j] * pressure[j];
            if (recurrent && use_hidden)
                for (size_t j = 0; j < hidden; ++j)
                    value += head_event[row + event_size + pressure.size() + j] * state.hidden[j];
            projection[unit] = value;
        }

        std::array<int32_t, max_candidates> result{};
        for (size_t i = 0; i < event.count; ++i) {
            const auto& candidate = event.candidates[i];
            auto source = compact_candidate_features(candidate, event.pc);
            const auto source_pc = lookup(candidate.source_pc);
            std::copy(source_pc.begin(), source_pc.end(), source.begin());
            float score = output_bias;
            for (size_t unit = 0; unit < projection.size(); ++unit) {
                float value = 0;
                for (size_t j = 0; j < source.size(); ++j)
                    value += head_source[unit * candidate_size + j] * source[j];
                score += reduction[unit] * std::max(0.0f, projection[unit] + value);
            }
            result[i] = fixed_score(score, 1024);
        }
        return result;
    }

private:
    using Embedding = std::array<float, 2>;

    struct PcEmbedding {
        uint32_t tag;
        Embedding values;
    };

    static float weight(std::ifstream& input)
    {
        float value;
        if (!(input >> value) || !std::isfinite(value))
            throw std::runtime_error("missing or invalid pressure model weight");
        return value;
    }

    static void load(std::ifstream& input, std::vector<float>& values, size_t count)
    {
        values.resize(count);
        for (auto& value : values)
            value = weight(input);
    }

    const Embedding& lookup(uint64_t pc) const
    {
        const auto tag = static_cast<uint32_t>(pc);
        const auto found = std::lower_bound(embeddings.begin(), embeddings.end(), tag,
                                            [](const PcEmbedding& entry, uint32_t key) { return entry.tag < key; });
        return found != embeddings.end() && found->tag == tag ? found->values : unknown;
    }

    size_t context_size() const { return event_size + 4 + hidden; }

    static float sigmoid(float value) { return 1 / (1 + std::exp(-value)); }

    void update(const EventFeatures& current, PressureState& state) const
    {
        if (reset_events && state.events == reset_events)
            state = {};
        std::array<float, 4 * pressure_max_hidden> gates;
        for (size_t row = 0; row < 4 * hidden; ++row) {
            float value = cell_bias[row];
            for (size_t j = 0; j < current.size(); ++j)
                value += weights_input[row * event_size + j] * current[j];
            for (size_t j = 0; j < hidden; ++j)
                value += weights_hidden[row * hidden + j] * state.hidden[j];
            gates[row] = value;
        }
        for (size_t j = 0; j < hidden; ++j) {
            const auto input = sigmoid(gates[j]);
            const auto forget = sigmoid(gates[hidden + j]);
            const auto candidate = std::tanh(gates[2 * hidden + j]);
            const auto output = sigmoid(gates[3 * hidden + j]);
            state.cell[j] = forget * state.cell[j] + input * candidate;
            state.hidden[j] = output * std::tanh(state.cell[j]);
        }
        ++state.events;
    }

    bool recurrent = false;
    size_t hidden;
    uint64_t reset_events;
    Embedding unknown{};
    std::vector<PcEmbedding> embeddings;
    std::vector<float> weights_input;
    std::vector<float> weights_hidden;
    std::vector<float> cell_bias;
    std::vector<float> head_event;
    std::vector<float> head_bias;
    std::vector<float> head_source;
    std::vector<float> reduction;
    float output_bias;
};

} // namespace llc_learned

#endif
