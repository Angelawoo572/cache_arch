#ifndef LLC_LEARNED_PREFETCH_H
#define LLC_LEARNED_PREFETCH_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llc_learned {

constexpr size_t event_size = 7;
constexpr size_t candidate_size = 6;
constexpr size_t hidden_size = 8;
constexpr size_t rank_size = 4;
constexpr size_t directory_size = 32;
constexpr size_t first_decision = 128;
constexpr size_t max_candidates = 16;
constexpr uint64_t page_lines = 64;
constexpr std::array<int, 10> offsets{{1, 2, -1, 3, -2, 4, -3, 6, -4, -6}};

using EventFeatures = std::array<float, event_size>;
using CandidateFeatures = std::array<float, candidate_size>;
using ModelVector = std::array<float, hidden_size>;

struct Candidate {
    uint64_t line;
    uint64_t source_pc;
    uint64_t source_offset;
    int delta;
    size_t age;
};

struct Observation {
    uint64_t line;
    uint64_t pc;
    EventFeatures features;
    std::array<Candidate, max_candidates> candidates;
    size_t count;
};

struct State {
    ModelVector hidden{};
    ModelVector cell{};
    uint64_t events = 0;
};

inline float low_pc(uint64_t pc)
{
    return static_cast<float>(pc & 255) / 255;
}

inline float high_pc(uint64_t pc)
{
    return static_cast<float>((pc >> 8) & 255) / 255;
}

class History {
public:
    Observation observe(uint64_t line, uint64_t pc)
    {
        Observation result{};
        result.line = line;
        result.pc = pc;
        const auto last = prior(1);
        const auto offset = static_cast<int>(line % page_lines);
        const bool same_page = used && line / page_lines == last.line / page_lines;
        const int delta = same_page ? offset - static_cast<int>(last.line % page_lines) : 0;
        result.features = {{low_pc(pc), high_pc(pc), static_cast<float>(offset) / 63,
                            static_cast<float>(std::max(-8, std::min(8, delta))) / 8,
                            static_cast<float>(!same_page), static_cast<float>(used && pc == last.pc),
                            static_cast<float>(same_page && delta == 0)}};
        if (observations >= first_decision)
            candidates(result);
        entries[next] = {line, pc};
        next = (next + 1) % entries.size();
        used = std::min(used + 1, entries.size());
        ++observations;
        return result;
    }

private:
    struct Demand {
        uint64_t line;
        uint64_t pc;
    };

    const Demand& prior(size_t age) const
    {
        return entries[(next + entries.size() - age) % entries.size()];
    }

    void candidates(Observation& result) const
    {
        std::array<std::array<size_t, 4>, 8> bands{};
        std::array<size_t, 8> counts{};
        const auto current_page = result.line / page_lines;
        for (size_t age = 1; age <= entries.size(); ++age) {
            const auto page = prior(age).line / page_lines;
            if (page == current_page)
                continue;
            bool newest = true;
            for (size_t other_age = 1; other_age < age; ++other_age) {
                if (prior(other_age).line / page_lines == page) {
                    newest = false;
                    break;
                }
            }
            if (newest) {
                const auto band = (age - 1) / 4;
                bands[band][counts[band]++] = age;
            }
        }
        constexpr std::array<size_t, 8> band_order{{3, 2, 1, 0, 4, 5, 6, 7}};
        for (const auto delta : offsets) {
            for (size_t round = 0; round < 4; ++round) {
                for (const auto band : band_order) {
                    if (round >= counts[band])
                        continue;
                    const auto age = bands[band][round];
                    const auto source = prior(age);
                    const auto source_offset = source.line % page_lines;
                    const int offset = static_cast<int>(source_offset) + delta;
                    if (offset < 0 || offset >= static_cast<int>(page_lines))
                        continue;
                    const auto target = (source.line / page_lines) * page_lines + offset;
                    if (!target || std::any_of(entries.begin(), entries.end(), [target](const Demand& previous) {
                            return previous.line == target;
                        }))
                        continue;
                    result.candidates[result.count++] = {target, source.pc, source_offset, delta, age};
                    if (result.count == result.candidates.size())
                        return;
                }
            }
        }
    }

    std::array<Demand, directory_size> entries{};
    size_t next = 0;
    size_t used = 0;
    uint64_t observations = 0;
};

inline CandidateFeatures candidate_features(const Candidate& candidate, uint64_t pc)
{
    return {{low_pc(candidate.source_pc), high_pc(candidate.source_pc),
             static_cast<float>(candidate.source_offset) / 63,
             static_cast<float>(candidate.delta) / 6, static_cast<float>(candidate.age) / 32,
             static_cast<float>(candidate.source_pc == pc)}};
}

inline int32_t fixed_score(float score, float scale)
{
    const float fixed = std::floor(score * scale + 0.5f);
    return static_cast<int32_t>(std::max(-32768.0f, std::min(32767.0f, fixed)));
}

class Model {
public:
    Model(const char* path, const std::string& expected_mode)
    {
        if (!path)
            throw std::runtime_error("missing model path");
        std::ifstream file(path);
        std::string magic, mode;
        size_t d = 0, h = 0, r = 0, f = 0, c = 0;
        if (!(file >> magic >> mode >> d >> h >> r >> f >> c >> reset_events)
            || magic != "LLC_LSTM_V1" || mode != expected_mode || d != event_size
            || h != hidden_size || r != rank_size || f != candidate_size || c != max_candidates)
            throw std::runtime_error("invalid or incompatible LLC LSTM model header");
        recurrent = mode == "lstm";
        read(file, weights_input);
        read(file, weights_hidden);
        read(file, cell_bias);
        read(file, weights_query);
        read(file, query_bias);
        read(file, weights_key);
        read(file, key_bias);
        read(file, weights_bias);
        read(file, candidate_bias);
        std::string trailing;
        if (file >> trailing)
            throw std::runtime_error("trailing tokens in LLC LSTM model");
    }

    std::array<int32_t, max_candidates> score(const Observation& event, State& state) const
    {
        if (recurrent)
            update(event.features, state);
        std::array<float, rank_size> query = query_bias;
        for (size_t rank = 0; rank < rank_size; ++rank) {
            const size_t begin = rank * (event_size + hidden_size);
            for (size_t i = 0; i < event_size; ++i)
                query[rank] += weights_query[begin + i] * event.features[i];
            for (size_t i = 0; i < hidden_size; ++i)
                query[rank] += weights_query[begin + event_size + i] * state.hidden[i];
        }
        std::array<int32_t, max_candidates> scores{};
        for (size_t i = 0; i < event.count; ++i) {
            const auto features = candidate_features(event.candidates[i], event.pc);
            float score = candidate_bias[0];
            for (size_t j = 0; j < candidate_size; ++j)
                score += weights_bias[j] * features[j];
            for (size_t rank = 0; rank < rank_size; ++rank) {
                float key = key_bias[rank];
                for (size_t j = 0; j < candidate_size; ++j)
                    key += weights_key[rank * candidate_size + j] * features[j];
                score += query[rank] * key;
            }
            scores[i] = fixed_score(score, 1024);
        }
        return scores;
    }

private:
    template <size_t N>
    static void read(std::ifstream& file, std::array<float, N>& values)
    {
        for (auto& value : values) {
            if (!(file >> value) || !std::isfinite(value))
                throw std::runtime_error("missing or invalid LLC LSTM model weight");
        }
    }

    static float sigmoid(float value)
    {
        return 1 / (1 + std::exp(-value));
    }

    void update(const EventFeatures& features, State& state) const
    {
        if (reset_events && state.events == reset_events)
            state = {};
        auto gates = cell_bias;
        for (size_t row = 0; row < gates.size(); ++row) {
            for (size_t i = 0; i < event_size; ++i)
                gates[row] += weights_input[row * event_size + i] * features[i];
            for (size_t i = 0; i < hidden_size; ++i)
                gates[row] += weights_hidden[row * hidden_size + i] * state.hidden[i];
        }
        for (size_t i = 0; i < hidden_size; ++i) {
            const float input = sigmoid(gates[i]);
            const float forget = sigmoid(gates[hidden_size + i]);
            const float proposed = std::tanh(gates[hidden_size * 2 + i]);
            const float output = sigmoid(gates[hidden_size * 3 + i]);
            state.cell[i] = forget * state.cell[i] + input * proposed;
            state.hidden[i] = output * std::tanh(state.cell[i]);
        }
        ++state.events;
    }

    bool recurrent = false;
    uint64_t reset_events = 0;
    std::array<float, hidden_size * 4 * event_size> weights_input{};
    std::array<float, hidden_size * 4 * hidden_size> weights_hidden{};
    std::array<float, hidden_size * 4> cell_bias{};
    std::array<float, rank_size * (event_size + hidden_size)> weights_query{};
    std::array<float, rank_size> query_bias{};
    std::array<float, rank_size * candidate_size> weights_key{};
    std::array<float, rank_size> key_bias{};
    std::array<float, candidate_size> weights_bias{};
    std::array<float, 1> candidate_bias{};
};

class PagePc {
public:
    explicit PagePc(const char* path)
    {
        if (!path)
            throw std::runtime_error("missing page-PC model path");
        std::ifstream file(path);
        std::string header;
        size_t width;
        if (!(file >> header) || header != "LLC_PAGE_PC_V1")
            throw std::runtime_error("invalid page-PC model header");
        while (file >> width) {
            Key key{};
            if (width != 2 && width != 5 && width != 6)
                throw std::runtime_error("invalid page-PC key width");
            key[0] = width;
            for (size_t i = 1; i <= width; ++i) {
                if (!(file >> key[i]))
                    throw std::runtime_error("truncated page-PC key");
            }
            Counts counts{};
            if (!(file >> counts.positive >> counts.total) || !counts.total || counts.positive > counts.total
                || !table.emplace(key, counts).second)
                throw std::runtime_error("invalid or duplicate page-PC counts");
        }
        if (!file.eof() || table.empty())
            throw std::runtime_error("invalid or empty page-PC table");
    }

    std::array<int32_t, max_candidates> score(const Observation& event) const
    {
        std::array<int32_t, max_candidates> scores{};
        for (size_t i = 0; i < event.count; ++i) {
            const auto& c = event.candidates[i];
            unsigned bucket = 0;
            for (size_t lag = c.age - 1; lag; lag >>= 1)
                ++bucket;
            const Key key{{6, c.delta, bucket, static_cast<int64_t>(event.pc),
                           static_cast<int64_t>(event.line % page_lines / 16),
                           static_cast<int64_t>(c.source_offset / 16), static_cast<int64_t>(c.source_pc)}};
            double probability = 0.005;
            for (const size_t width : {2, 5, 6}) {
                Key partial{};
                std::copy_n(key.begin() + 1, width, partial.begin() + 1);
                partial[0] = width;
                const auto found = table.find(partial);
                if (found != table.end())
                    probability = (found->second.positive + 20 * probability) / (found->second.total + 20);
            }
            scores[i] = fixed_score(static_cast<float>(std::log(probability / (1 - probability))), 1024);
        }
        return scores;
    }

private:
    using Key = std::array<int64_t, 7>;
    struct Counts {
        uint64_t positive;
        uint64_t total;
    };
    struct Hash {
        size_t operator()(const Key& key) const
        {
            uint64_t hash = 14695981039346656037ull;
            for (const auto item : key) {
                hash ^= static_cast<uint64_t>(item);
                hash *= 1099511628211ull;
            }
            return static_cast<size_t>(hash);
        }
    };
    std::unordered_map<Key, Counts, Hash> table;
};

} // namespace llc_learned
#endif
