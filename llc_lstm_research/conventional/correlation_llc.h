#ifndef LLC_CORRELATION_H
#define LLC_CORRELATION_H

#include "learned_llc.h"

#include <iterator>
#include <list>
#include <tuple>

namespace llc_learned {

class LlcCorrelationTable {
public:
    struct Offset {
        int8_t delta;
        uint8_t strength;
    };
    using Row = std::array<Offset, 4>;

    explicit LlcCorrelationTable(const char* path)
    {
        std::ifstream input(path);
        std::string version;
        if (!(input >> version) || version != "LLC_V2_CORRELATION_V1")
            throw std::runtime_error("invalid address correlation table");
        int level;
        while (input >> level) {
            unsigned current_pc, current_bucket, source_pc, source_bucket, lag, relation, strength;
            int delta;
            if (!(input >> current_pc >> current_bucket >> source_pc >> source_bucket >> lag
                    >> relation >> delta >> strength)
                || level < 0 || level > 2 || current_pc > 65535 || source_pc > 65535
                || current_bucket > 3 || source_bucket > 3 || lag > 5 || relation > 1
                || delta < -63 || delta > 63 || !strength || strength > 255)
                throw std::runtime_error("invalid entry in address correlation table");
            auto& row = table[key(level, current_pc, current_bucket, source_pc, source_bucket, lag, relation)];
            const auto empty = std::find_if(row.begin(), row.end(), [](Offset value) { return !value.strength; });
            if (empty == row.end() || std::any_of(row.begin(), empty, [delta](Offset value) { return value.delta == delta; }))
                throw std::runtime_error("duplicate or oversize address correlation row");
            *empty = {static_cast<int8_t>(delta), static_cast<uint8_t>(strength)};
        }
        if (!input.eof() || table.empty())
            throw std::runtime_error("malformed or empty address correlation table");
    }

    const Row& lookup(uint32_t current_pc, unsigned current_bucket, uint32_t source_pc,
                      unsigned source_bucket, unsigned lag, unsigned relation) const
    {
        const auto exact = table.find(key(2, current_pc, current_bucket, source_pc, source_bucket, lag, relation));
        if (exact != table.end())
            return exact->second;
        const auto source = table.find(key(1, 0, 0, source_pc, source_bucket, 0, relation));
        if (source != table.end())
            return source->second;
        const auto common = table.find(key(0, 0, 0, 0, 0, 0, relation));
        return common != table.end() ? common->second : empty_row;
    }

private:
    static uint64_t key(unsigned level, unsigned current_pc, unsigned current_bucket,
                        unsigned source_pc, unsigned source_bucket, unsigned lag, unsigned relation)
    {
        return static_cast<uint64_t>(level) << 40
            | static_cast<uint64_t>(current_pc & 65535) << 24
            | static_cast<uint64_t>(source_pc & 65535) << 8
            | current_bucket << 6 | source_bucket << 4 | lag << 1 | relation;
    }

    const Row empty_row{};
    std::unordered_map<uint64_t, Row> table;
};

class LlcCorrelationHistory {
public:
    Observation observe(uint64_t line, uint64_t pc, uint64_t cycle, const LlcCorrelationTable& table,
                        uint64_t cooldown = 0, size_t cooldown_capacity = 0)
    {
        const Demand current{line, static_cast<uint32_t>(pc), static_cast<uint32_t>(cycle)};
        const auto previous = prior(1);
        const auto offset = static_cast<int>(line % page_lines);
        const bool same_page = used && line / page_lines == previous.line / page_lines;
        const int delta = same_page ? offset - static_cast<int>(previous.line % page_lines) : 0;
        Observation result{};
        result.line = line;
        result.pc = pc;
        result.features = {{low_pc(pc), high_pc(pc), static_cast<float>(offset) / 63,
                            static_cast<float>(std::max(-8, std::min(8, delta))) / 8,
                            static_cast<float>(!same_page), static_cast<float>(used && current.pc == previous.pc),
                            used ? elapsed_feature(current.tick - previous.tick) : 0}};
        if (observations >= first_decision)
            candidates(result, current, table, cycle, cooldown, cooldown_capacity);
        entries[next] = current;
        next = (next + 1) % entries.size();
        used = std::min(used + 1, entries.size());
        ++observations;
        return result;
    }

private:
    struct Demand {
        uint64_t line;
        uint32_t pc;
        uint32_t tick;
    };
    static_assert(sizeof(Demand) == 16, "address directory entry must stay at 16 bytes");

    struct Ranked {
        Candidate candidate;
        unsigned rank;
        unsigned strength;
    };

    struct LastProposal {
        uint64_t cycle;
        std::list<uint64_t>::iterator order;
    };

    const Demand& prior(size_t age) const
    {
        return entries[(next + entries.size() - age) % entries.size()];
    }

    static bool less_candidate(const Ranked& left, const Ranked& right)
    {
        const auto order = [](const Ranked& value) {
            return std::make_tuple(value.rank, 255u - value.strength, value.candidate.age,
                                   value.candidate.delta, value.candidate.line);
        };
        return order(left) < order(right);
    }

    void cooldown_shortlist(Observation& observation, Ranked* first, Ranked* end, uint64_t cycle,
                            uint64_t cooldown, size_t capacity)
    {
        std::sort(first, end, less_candidate);
        std::array<const Candidate*, max_candidates> repeats;
        size_t repeat_count = 0;
        for (auto proposal = first; proposal != end && observation.count < max_candidates; ++proposal) {
            const auto line = proposal->candidate.line;
            const auto previous = last_proposal.find(line);
            if (previous == last_proposal.end() || cycle - previous->second.cycle >= cooldown)
                observation.candidates[observation.count++] = proposal->candidate;
            else if (repeat_count < repeats.size())
                repeats[repeat_count++] = &proposal->candidate;
        }
        // Repeats used as fallback do not refresh the cooldown or LRU order.
        for (size_t i = 0; i < observation.count; ++i) {
            const auto line = observation.candidates[i].line;
            const auto previous = last_proposal.find(line);
            if (previous != last_proposal.end())
                proposal_order.erase(previous->second.order);
            proposal_order.push_back(line);
            last_proposal[line] = {cycle, std::prev(proposal_order.end())};
        }
        while (last_proposal.size() > capacity) {
            last_proposal.erase(proposal_order.front());
            proposal_order.pop_front();
        }
        for (size_t i = 0; i < repeat_count && observation.count < max_candidates; ++i)
            observation.candidates[observation.count++] = *repeats[i];
    }

    void candidates(Observation& observation, const Demand& current, const LlcCorrelationTable& table,
                    uint64_t cycle, uint64_t cooldown, size_t cooldown_capacity)
    {
        std::array<Ranked, 4 * (directory_size + 1)> proposals{};
        size_t proposed = 0;
        std::array<uint64_t, directory_size + 1> pages{};
        size_t page_count = 0;
        for (size_t age = 0; age <= used; ++age) {
            const auto source = age ? prior(age) : current;
            const uint64_t page = source.line / page_lines;
            if (std::find(pages.begin(), pages.begin() + page_count, page) != pages.begin() + page_count)
                continue;
            pages[page_count++] = page;
            const unsigned source_offset = source.line % page_lines;
            unsigned lag = 0;
            for (size_t value = age ? age - 1 : 0; value; value >>= 1)
                ++lag;
            const auto& row = table.lookup(current.pc, current.line % page_lines / 16, source.pc,
                                           source_offset / 16, lag, age != 0);
            for (unsigned rank = 0; rank < row.size() && row[rank].strength; ++rank) {
                const int delta = row[rank].delta;
                const int offset = static_cast<int>(source_offset) + delta;
                if (offset < 0 || offset >= static_cast<int>(page_lines))
                    continue;
                const auto line = page * page_lines + offset;
                if (!line || line == current.line ||
                    std::any_of(entries.begin(), entries.end(), [line](const Demand& demand) { return demand.line == line; }))
                    continue;
                const auto elapsed = std::min(current.tick - source.tick, uint32_t{4095});
                proposals[proposed++] = {
                    {line, source.pc, source_offset, delta, static_cast<uint16_t>(age), static_cast<uint16_t>(elapsed)},
                    rank, row[rank].strength
                };
            }
        }
        if (cooldown) {
            cooldown_shortlist(observation, proposals.data(), proposals.data() + proposed, cycle, cooldown,
                               cooldown_capacity);
            return;
        }
        observation.count = std::min(proposed, max_candidates);
        std::partial_sort(proposals.begin(), proposals.begin() + observation.count, proposals.begin() + proposed,
                          less_candidate);
        for (size_t i = 0; i < observation.count; ++i)
            observation.candidates[i] = proposals[i].candidate;
    }

    std::array<Demand, directory_size> entries{};
    size_t next = 0;
    size_t used = 0;
    uint64_t observations = 0;
    std::list<uint64_t> proposal_order;
    std::unordered_map<uint64_t, LastProposal> last_proposal;
};

} // namespace llc_learned

#endif
