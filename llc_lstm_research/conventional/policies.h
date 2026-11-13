#ifndef LLC_CONVENTIONAL_POLICIES_H
#define LLC_CONVENTIONAL_POLICIES_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "champsim.h"
#include "ampm.h"
#include "memory_class.h"
#include "spp_dev2_helper.h"
#include "streamer.h"
#include "stride.h"

// Pythia v1.3 declares the destructor but leaves its process-lifetime AMPM objects alive.
inline AMPM::~AMPM()
{
    for (auto* entry : page_buffer)
        delete entry;
}

namespace llc_conventional {

enum class PolicyKind { none, next_line, stride, streamer, ampm, spp };

constexpr uint64_t lines_per_page = PAGE_SIZE / BLOCK_SIZE;

class SppPolicy {
public:
    std::vector<uint64_t> access(uint64_t line)
    {
        const uint64_t page = line / lines_per_page;
        const auto offset = static_cast<int>(line % lines_per_page);
        uint32_t last_signature = 0;
        uint32_t signature = 0;
        int32_t delta = 0;
        history.global_accuracy = history.pf_issued ? 100 * history.pf_useful / history.pf_issued : 0;
        signatures.read_and_update_sig(page, offset, last_signature, signature, delta, history);
        if (last_signature)
            patterns.update_pattern(last_signature, delta);

        struct Candidate {
            uint64_t line;
            uint32_t confidence;
        };
        std::vector<Candidate> predictions;
        int base = offset;
        uint32_t path_confidence = 100;
        uint32_t depth = 0;
        constexpr unsigned max_stages = 4;
        for (unsigned stage = 0; stage < max_stages; ++stage) {
            // Each native pattern lookup can return at most one delta per way.
            int deltas[PT_WAY];
            uint32_t confidence[PT_WAY];
            uint32_t best_way = PT_WAY;
            uint32_t count = 0;
            patterns.read_pattern(signature, deltas, confidence, best_way, path_confidence, count, depth, history);

            for (uint32_t index = 0; index < count; ++index) {
                const int target = base + deltas[index];
                if (!on_page(target)) {
                    const int page_lines = lines_per_page;
                    const auto next_offset = static_cast<uint32_t>((target % page_lines + page_lines) % page_lines);
                    history.update_entry(signature, confidence[index], next_offset, deltas[index]);
                    continue;
                }
                const uint64_t candidate = page * lines_per_page + target;
                if (candidate != line)
                    predictions.push_back({candidate, confidence[index]});
            }

            if (best_way == PT_WAY)
                break;
            const int step = patterns.delta[get_hash(signature) % PT_SET][best_way];
            base += step;
            if (!on_page(base))
                break;
            const uint32_t encoded_step = step < 0 ? -step + (1 << (SIG_DELTA_BIT - 1)) : step;
            signature = ((signature << SIG_SHIFT) ^ encoded_step) & SIG_MASK;
        }

        // Keep native discovery order when confidence ties under a shared degree cap.
        std::stable_sort(predictions.begin(), predictions.end(), [](const Candidate& a, const Candidate& b) {
            return a.confidence > b.confidence;
        });
        std::vector<uint64_t> candidates;
        for (const auto& prediction : predictions) {
            if (std::find(candidates.begin(), candidates.end(), prediction.line) == candidates.end())
                candidates.push_back(prediction.line);
        }
        return candidates;
    }

    void issued(uint64_t line)
    {
        auto& entry = feedback_for(line);
        entry.line = line;
        entry.state = State::pending;
        ++history.pf_issued;
        if (history.pf_issued > GLOBAL_COUNTER_MAX) {
            history.pf_issued >>= 1;
            history.pf_useful >>= 1;
        }
    }

    void demand_outcome(uint64_t line, bool hit)
    {
        auto& entry = feedback_for(line);
        if (entry.line != line || entry.state == State::empty)
            return;

        if (hit && entry.state == State::resident && history.pf_useful < history.pf_issued)
            ++history.pf_useful;
        // A demand that arrives before the prefetch fills cannot use it as an LLC hit.
        entry.state = State::empty;
    }

    void fill(uint64_t line, bool prefetch, uint64_t evicted_line, bool evicted_valid)
    {
        if (evicted_valid) {
            auto& victim = feedback_for(evicted_line);
            if (victim.line == evicted_line && victim.state == State::resident)
                victim.state = State::empty;
        }

        auto& entry = feedback_for(line);
        if (entry.line == line && entry.state == State::pending)
            entry.state = prefetch ? State::resident : State::empty;
    }

private:
    enum class State { empty, pending, resident };

    struct Feedback {
        uint64_t line = 0;
        State state = State::empty;
    };

    static bool on_page(int offset)
    {
        return offset >= 0 && offset < static_cast<int>(lines_per_page);
    }

    Feedback& feedback_for(uint64_t line)
    {
        return feedback[(get_hash(line) >> REMAINDER_BIT) % FILTER_SET];
    }

    SIGNATURE_TABLE signatures;
    PATTERN_TABLE patterns;
    GLOBAL_REGISTER history;
    // Keep the native SPP filter's capacity and indexing with full line tags for LLC feedback.
    std::array<Feedback, FILTER_SET> feedback{};
};

class Policy {
public:
    explicit Policy(PolicyKind kind) : kind(kind)
    {
        // Retain the derived deleter because Pythia's Prefetcher destructor is non-virtual.
        switch (kind) {
        case PolicyKind::stride:
            native = std::make_shared<StridePrefetcher>("stride");
            break;
        case PolicyKind::streamer:
            native = std::make_shared<Streamer>("streamer");
            break;
        case PolicyKind::ampm:
            native = std::make_shared<AMPM>("ampm");
            break;
        case PolicyKind::spp:
            spp.reset(new SppPolicy);
            break;
        case PolicyKind::none:
        case PolicyKind::next_line:
            break;
        }
    }

    std::vector<uint64_t> access(uint64_t line, uint64_t pc)
    {
        if (spp)
            return spp->access(line);
        if (kind == PolicyKind::next_line && line % lines_per_page + 1 < lines_per_page)
            return {line + 1};
        if (!native)
            return {};

        std::vector<uint64_t> candidates;
        native->invoke_prefetcher(pc, line << LOG2_BLOCK_SIZE, 0, LOAD, candidates);
        for (auto& candidate : candidates)
            candidate >>= LOG2_BLOCK_SIZE;
        // Stock Stride includes the triggering line as its zero-degree candidate.
        candidates.erase(std::remove(candidates.begin(), candidates.end(), line), candidates.end());
        return candidates;
    }

    void issued(uint64_t line)
    {
        if (spp)
            spp->issued(line);
    }

    void demand_outcome(uint64_t line, bool hit)
    {
        if (spp)
            spp->demand_outcome(line, hit);
    }

    void fill(uint64_t line, bool prefetch, uint64_t evicted_line, bool evicted_valid)
    {
        if (spp)
            spp->fill(line, prefetch, evicted_line, evicted_valid);
    }

private:
    PolicyKind kind;
    std::shared_ptr<Prefetcher> native;
    std::unique_ptr<SppPolicy> spp;
};

} // namespace llc_conventional

#endif
