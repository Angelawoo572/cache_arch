#ifndef LLC_POSITION_PREFETCH_H
#define LLC_POSITION_PREFETCH_H

#include "learned_llc.h"

namespace llc_learned {

constexpr size_t max_position_slots = 4;

struct PositionSource {
    uint64_t line;
    uint32_t pc;
    uint32_t gap;
    uint16_t age;
    uint64_t allowed;
};

struct PositionObservation {
    std::array<PositionSource, max_position_slots> sources{};
    size_t count = 0;
};

inline float position_gap(uint32_t cycles)
{
    uint32_t n = std::min<uint32_t>(cycles, 4095) + 1;
    uint32_t power = 0;
    while (n > 1) {
        n >>= 1;
        ++power;
    }
    return static_cast<float>(power) / 12;
}

class PositionHistory {
public:
    PositionObservation observe(Observation& event, uint64_t line, uint64_t pc, uint64_t cycle,
                                size_t slots)
    {
        const Demand current{line, static_cast<uint32_t>(pc), static_cast<uint32_t>(cycle)};
        const auto& previous = prior(1);
        const auto offset = static_cast<int>(line % page_lines);
        const bool same_page = length && line / page_lines == previous.line / page_lines;
        const int delta = same_page ? offset - static_cast<int>(previous.line % page_lines) : 0;
        event.line = line;
        event.pc = pc;
        event.features = {{low_pc(pc), high_pc(pc), static_cast<float>(offset) / 63,
                           static_cast<float>(std::max(-8, std::min(8, delta))) / 8,
                           static_cast<float>(!same_page), static_cast<float>(length && current.pc == previous.pc),
                           length ? position_gap(current.cycle - previous.cycle) : 0}};

        PositionObservation result;
        if (observations >= first_decision) {
            append_source(result, current, current, 0);
            for (size_t age = 1; age <= length && result.count < slots; ++age) {
                const auto& source = prior(age);
                const auto page = source.line / page_lines;
                auto matches_page = [page](const PositionSource& item) {
                    return item.line / page_lines == page;
                };
                const auto end = result.sources.begin() + result.count;
                if (std::none_of(result.sources.begin(), end, matches_page))
                    append_source(result, current, source, age);
            }
        }

        entries[next] = current;
        next = (next + 1) % entries.size();
        length = std::min(length + 1, entries.size());
        ++observations;
        return result;
    }

private:
    struct Demand {
        uint64_t line;
        uint32_t pc;
        uint32_t cycle;
    };

    const Demand& prior(size_t age) const
    {
        return entries[(next + entries.size() - age) % entries.size()];
    }

    void append_source(PositionObservation& result, const Demand& current, const Demand& source,
                       size_t age) const
    {
        const auto page = source.line / page_lines;
        uint64_t allowed = page ? UINT64_MAX : (UINT64_MAX - 1);
        if (!age)
            allowed &= ~(uint64_t{1} << (current.line % page_lines));
        for (size_t i = 1; i <= length; ++i) {
            const auto line = prior(i).line;
            if (line / page_lines == page)
                allowed &= ~(uint64_t{1} << (line % page_lines));
        }
        result.sources[result.count++] = {source.line, source.pc, current.cycle - source.cycle,
                                          static_cast<uint16_t>(age), allowed};
    }

    std::array<Demand, directory_size> entries{};
    size_t next = 0;
    size_t length = 0;
    size_t observations = 0;
};

class PositionModel {
public:
    PositionModel(const char* path, const std::string& policy)
    {
        if (!path)
            throw std::runtime_error("position scoring requires a model file");
        std::ifstream input(path);
        std::string header, kind;
        unsigned relative, discrete;
        if (!(input >> header >> kind >> slots >> relative >> discrete) || header != "LLC_POSITION_V1"
            || kind != policy || (kind != "event" && kind != "lstm") || (slots != 1 && slots != 2 && slots != 4)
            || relative > 1 || discrete > 1)
            throw std::runtime_error("invalid position model metadata");
        recurrent = kind == "lstm";
        relative_position = relative;
        discrete_pc = discrete;
        const size_t extra = discrete_pc ? embedding_width - 2 : 0;
        event_width = event_size + extra;
        source_width = candidate_size + extra;
        query_width = event_width + (recurrent ? hidden_size : 0);
        const size_t positions = relative_position ? 2 * page_lines - 1 : page_lines;
        if (discrete_pc)
            load(input, embedding, 64 * embedding_width);
        if (recurrent) {
            load(input, input_weights, 4 * hidden_size * event_width);
            load(input, hidden_weights, 4 * hidden_size * hidden_size);
            load(input, gate_bias, 4 * hidden_size);
        }
        load(input, query_weights, hidden_size * query_width);
        load(input, query_bias, hidden_size);
        load(input, source_weights, hidden_size * source_width);
        load(input, position_weights, positions * hidden_size);
        load(input, position_bias, positions);
        std::string extra_token;
        if (input >> extra_token || !input.eof())
            throw std::runtime_error("unexpected trailing position model data");
    }

    size_t source_slots() const { return slots; }

    std::array<int32_t, max_candidates> score(Observation& event, const PositionObservation& position,
                                              State& state) const
    {
        std::array<float, event_size + embedding_width - 2 + hidden_size> context{};
        encode_pc(event.features, 0, event.pc, context);
        if (recurrent) {
            update_state(context.data(), state);
            std::copy(state.hidden.begin(), state.hidden.end(), context.begin() + event_width);
        }
        ModelVector query;
        for (size_t i = 0; i < hidden_size; ++i)
            query[i] = query_bias[i] + dot(query_weights, i, query_width, context.data());

        struct ScoredLine {
            Candidate candidate;
            int32_t score;
            size_t order;
        };
        std::array<ScoredLine, max_position_slots * page_lines> choices;
        size_t count = 0;
        for (size_t slot = 0; slot < position.count; ++slot) {
            const auto& source = position.sources[slot];
            const size_t origin = source.line % page_lines;
            CandidateFeatures raw{{static_cast<float>(origin) / 63, low_pc(source.pc), high_pc(source.pc),
                                   static_cast<float>(!source.age), static_cast<float>(source.pc == uint32_t(event.pc)),
                                   position_gap(source.gap)}};
            std::array<float, candidate_size + embedding_width - 2> features;
            encode_pc(raw, 1, source.pc, features);
            ModelVector activation;
            for (size_t i = 0; i < hidden_size; ++i)
                activation[i] = std::max(0.0f, query[i] + dot(source_weights, i, source_width, features.data()));
            for (size_t offset = 0; offset < page_lines; ++offset) {
                if (!(source.allowed & (uint64_t{1} << offset)))
                    continue;
                const int delta = static_cast<int>(offset) - static_cast<int>(origin);
                const size_t output = relative_position ? static_cast<int>(page_lines - 1) + delta : offset;
                const float logit = position_bias[output] + dot(position_weights, output, hidden_size, activation.data());
                const auto quantized = std::min(32767.0f, std::max(-32768.0f, std::floor(1024 * logit + .5f)));
                choices[count++] = {{source.line - origin + offset, source.pc, origin, delta, source.age,
                                     static_cast<uint16_t>(std::min<uint32_t>(source.gap, 65535))},
                                    static_cast<int32_t>(quantized), slot * page_lines + offset};
            }
        }
        event.count = std::min(count, max_candidates);
        std::partial_sort(choices.begin(), choices.begin() + event.count, choices.begin() + count,
                          [](const ScoredLine& a, const ScoredLine& b) {
                              return a.score != b.score ? a.score > b.score : a.order < b.order;
                          });
        std::array<int32_t, max_candidates> scores{};
        for (size_t i = 0; i < event.count; ++i) {
            event.candidates[i] = choices[i].candidate;
            scores[i] = choices[i].score;
        }
        return scores;
    }

private:
    static constexpr size_t embedding_width = 4;

    static void load(std::ifstream& input, std::vector<float>& tensor, size_t size)
    {
        tensor.resize(size);
        for (auto& value : tensor)
            if (!(input >> value) || !std::isfinite(value))
                throw std::runtime_error("missing or invalid position model coefficient");
    }

    static float dot(const std::vector<float>& weights, size_t row, size_t width, const float* input)
    {
        float result = 0;
        for (size_t column = 0; column < width; ++column)
            result += weights[row * width + column] * input[column];
        return result;
    }

    template <size_t Input, size_t Output>
    void encode_pc(const std::array<float, Input>& raw, size_t at, uint64_t pc,
                   std::array<float, Output>& output) const
    {
        if (!discrete_pc) {
            std::copy(raw.begin(), raw.end(), output.begin());
            return;
        }
        const auto low_pc = pc & 65535;
        const auto index = ((low_pc >> 2) ^ (low_pc >> 9)) & 63;
        std::copy_n(raw.begin(), at, output.begin());
        std::copy_n(embedding.begin() + index * embedding_width, embedding_width, output.begin() + at);
        std::copy(raw.begin() + at + 2, raw.end(), output.begin() + at + embedding_width);
    }

    static float sigmoid(float x)
    {
        return 1 / (1 + std::exp(-x));
    }

    void update_state(const float* event, State& state) const
    {
        std::array<float, 4 * hidden_size> gates;
        for (size_t i = 0; i < gates.size(); ++i)
            gates[i] = gate_bias[i] + dot(input_weights, i, event_width, event)
                        + dot(hidden_weights, i, hidden_size, state.hidden.data());
        for (size_t i = 0; i < hidden_size; ++i) {
            state.cell[i] = sigmoid(gates[i + hidden_size]) * state.cell[i]
                            + sigmoid(gates[i]) * std::tanh(gates[i + 2 * hidden_size]);
            state.hidden[i] = sigmoid(gates[i + 3 * hidden_size]) * std::tanh(state.cell[i]);
        }
        ++state.events;
    }

    size_t slots = 0;
    bool recurrent = false;
    bool relative_position = false;
    bool discrete_pc = false;
    size_t event_width = 0;
    size_t source_width = 0;
    size_t query_width = 0;
    std::vector<float> embedding, input_weights, hidden_weights, gate_bias;
    std::vector<float> query_weights, query_bias, source_weights, position_weights, position_bias;
};

}

#endif
