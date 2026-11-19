// Join the actual LLC admission sequence to outcomes on the identical dynamic operand.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>

using U = uint64_t;
using S = int64_t;
U le64(const unsigned char *v) {
    U word = 0;
    for (int i = 7; i >= 0; --i)
        word = (word << 8) | v[i];
    return word;
}
void read_exact(gzFile f, unsigned char *dst, unsigned n) {
    for (unsigned pos = 0; pos < n;) {
        int got = gzread(f, dst + pos, n - pos);
        if (got <= 0)
            throw std::runtime_error("truncated output stream");
        pos += got;
    }
}

// LLC arrival can reorder loads within the live processor window. Two retained blocks
// allow a backwards reference without expanding the entire 400-MB outcome stream.
struct Outcomes {
    static constexpr U block = 65536;
    gzFile file;
    U words, next = 0;
    S tags[2] = {-1, -1};
    std::array<std::vector<unsigned char>, 2> buf;
    Outcomes(const char *path, U n) : file(gzopen(path, "rb")), words(n) {
        if (!file)
            throw std::runtime_error("cannot open dynamic outcomes");
        for (auto &v : buf)
            v.resize(2 * block);
    }
    unsigned get(U pos, unsigned op) {
        if (pos >= words)
            throw std::runtime_error("event outside measured window");
        U b = pos / block;
        while (next <= b) {
            auto idx = next % 2;
            read_exact(file, buf[idx].data(), 2 * std::min(block, words - next * block));
            tags[idx] = next++;
        }
        if (tags[b % 2] != S(b))
            throw std::runtime_error("admission exceeds alignment lookback");
        auto ptr = &buf[b % 2][2 * (pos % block)];
        return ((unsigned(ptr[0]) | (unsigned(ptr[1]) << 8)) >> (4 * op)) & 15;
    }
    ~Outcomes() {
        if (file)
            gzclose(file);
    }
};
constexpr U Outcomes::block;

struct Event {
    U idop, pc, line, cycle;
    uint32_t rank;
    unsigned char baseline, arm;
};
using Sequence = std::vector<const Event *>;
struct Group {
    const Event *prev = nullptr, *last_timely = nullptr;
    U size = 0, hits = 0;
};
struct Gap {
    std::vector<S> cycles, ranks;
    void add(const Event &a, const Event &b) {
        cycles.push_back(S(b.cycle) - S(a.cycle));
        ranks.push_back(S(b.rank) - S(a.rank));
    }
    void append(const Gap &g) {
        cycles.insert(cycles.end(), g.cycles.begin(), g.cycles.end());
        ranks.insert(ranks.end(), g.ranks.begin(), g.ranks.end());
    }
};
enum Field {
    events,
    groups,
    never_groups,
    never_events,
    start_direct,
    start_no_direct,
    end_direct,
    nopf_misses,
    direct_hits,
    ordinary_hits,
    late,
    persistent,
    no_tag,
    adjacent_from_direct,
    next_direct,
    next_ordinary,
    next_late,
    next_miss,
    next_no_tag,
    episode_late,
    episode_plain,
    clock_reversals,
    instruction_reversals,
    supported_groups,
    supported_events,
    supported_never_groups,
    supported_never_events,
    field_count
};
const char *headers =
    "events,group_instances,never_timely_group_instances,events_in_never_timely_groups,first_direct,first_"
    "not_direct,"
    "last_direct,nopf_misses,direct_hits,ordinary_hits,late_merges,plain_misses,no_tag,adjacent_after_direct,"
    "next_direct,next_ordinary,next_late,next_plain_miss,next_no_tag,first_late_after_hit_run,"
    "first_plain_miss_after_hit_run,adjacent_clock_reversals,adjacent_idop_reversals,"
    "groups_at_least_32,events_in_supported_groups,never_timely_supported_groups,events_in_never_timely_"
    "supported_groups";
struct Counts {
    std::array<U, field_count> n{};
    U previous[3][5] = {};
    Gap gap;
    void append(const Counts &x) {
        for (unsigned i = 0; i < field_count; ++i)
            n[i] += x.n[i];
        for (unsigned a = 0; a < 3; ++a)
            for (unsigned b = 0; b < 5; ++b)
                previous[a][b] += x.previous[a][b];
        gap.append(x.gap);
    }
};
Counts measure(const Sequence &seq, int scope, bool orig) {
    Counts stats;
    auto &n = stats.n;
    std::unordered_map<U, Group> groups_by_key;
    for (auto p : seq) {
        if (orig && p->baseline != 2)
            continue;
        U key = scope == 0 ? 0 : scope == 1 ? p->pc : scope == 2 ? p->line / 64 : p->line / 4096;
        auto &g = groups_by_key[key];
        ++n[events];
        ++g.size;
        n[nopf_misses] += p->baseline == 2;
        const auto m = p->arm;
        if (m == 9) {
            ++n[direct_hits];
            ++g.hits;
        } else if (m == 1)
            ++n[ordinary_hits];
        else if (m == 6)
            ++n[late];
        else if (m == 2)
            ++n[persistent];
        else if (m == 0)
            ++n[no_tag];
        else
            throw std::runtime_error("invalid policy mask");
        unsigned before = !g.prev ? 0 : g.prev->arm == 9 ? 1 : 2;
        unsigned now = m == 9 ? 0 : m == 6 ? 1 : m == 2 ? 2 : m == 1 ? 3 : 4;
        ++stats.previous[before][now];
        if (!g.prev)
            ++n[m == 9 ? start_direct : start_no_direct];
        else {
            n[clock_reversals] += p->cycle < g.prev->cycle;
            n[instruction_reversals] += p->idop < g.prev->idop;
            if (g.prev->arm == 9) {
                ++n[adjacent_from_direct];
                ++n[m == 9   ? next_direct
                    : m == 1 ? next_ordinary
                    : m == 6 ? next_late
                    : m == 2 ? next_miss
                             : next_no_tag];
            }
        }
        if (m == 9)
            g.last_timely = p;
        else if (m != 1) {
            if (g.last_timely && (m == 2 || m == 6)) {
                ++n[m == 2 ? episode_plain : episode_late];
                stats.gap.add(*g.last_timely, *p);
            }
            g.last_timely = nullptr;
        }
        g.prev = p;
    }
    n[groups] = groups_by_key.size();
    for (const auto &kv : groups_by_key) {
        const auto &g = kv.second;
        if (g.hits == 0) {
            ++n[never_groups];
            n[never_events] += g.size;
        }
        if (g.size >= 32) {
            ++n[supported_groups];
            n[supported_events] += g.size;
            if (!g.hits) {
                ++n[supported_never_groups];
                n[supported_never_events] += g.size;
            }
        }
        n[end_direct] += g.prev->arm == 9;
    }
    if (n[adjacent_from_direct] + n[end_direct] != n[direct_hits] ||
        n[start_direct] + n[start_no_direct] != n[groups])
        throw std::runtime_error("run boundaries do not reconcile");
    return stats;
}

void emit(const std::string &key, Counts &counts, std::ofstream &counts_file, std::ofstream &gaps,
          std::ofstream &previous) {
    const char *before[] = {"first", "after_timely", "after_nondirect"};
    const char *now[] = {"timely_first_use", "late_first_use", "other_tag_miss", "ordinary_tag_hit",
                         "no_llc_tag"};
    for (unsigned a = 0; a < 3; ++a)
        for (unsigned b = 0; b < 5; ++b)
            previous << key << ',' << before[a] << ',' << now[b] << ',' << counts.previous[a][b] << '\n';
    counts_file << key;
    for (auto x : counts.n)
        counts_file << ',' << x;
    counts_file << '\n';
    for (bool cycles : {true, false}) {
        auto &v = cycles ? counts.gap.cycles : counts.gap.ranks;
        std::sort(v.begin(), v.end());
        auto n = v.size();
        gaps << key << ',' << (cycles ? "cycle_gap" : "ingress_rank_gap") << ',' << n << ','
             << std::distance(v.begin(), std::lower_bound(v.begin(), v.end(), S(0)));
        for (unsigned k : {10u, 50u, 90u}) {
            gaps << ',';
            if (n)
                gaps << v[(k * n + 99) / 100 - 1];
        }
        gaps << '\n';
    }
}

int main(int argc, char **argv) {
    if (argc != 8)
        return 2;
    const std::string case_id = argv[1], out = argv[7];
    const U start = std::stoull(argv[5]), roi = std::stoull(argv[6]);
    Outcomes nopf(argv[3], roi), stride(argv[4], roi);
    auto input = gzopen(argv[2], "rb");
    if (!input)
        throw std::runtime_error("cannot open ingress stream");
    std::vector<Event> log;
    std::array<unsigned char, 131072> bytes;
    for (;;) {
        int got = gzread(input, bytes.data(), bytes.size());
        if (!got && gzeof(input))
            break;
        if (got < 0 || got % 32 || !got)
            throw std::runtime_error("damaged ingress stream");
        for (int i = 0; i < got; i += 32) {
            const auto ptr = bytes.data() + i;
            U idop = le64(ptr), pos = (idop >> 2) - start;
            if (log.size() >= UINT32_MAX)
                throw std::runtime_error("ingress-rank overflow");
            Event e{idop,
                    le64(ptr + 8),
                    le64(ptr + 16),
                    le64(ptr + 24),
                    uint32_t(log.size()),
                    static_cast<unsigned char>(nopf.get(pos, idop & 3)),
                    static_cast<unsigned char>(stride.get(pos, idop & 3))};
            if ((!log.empty() && e.cycle < log.back().cycle) || e.baseline > 2)
                throw std::runtime_error("invalid arrival ordering or baseline mask");
            log.push_back(e);
        }
    }
    if (gzclose(input) != Z_OK)
        throw std::runtime_error("ingress stream failed integrity check");
    std::ofstream counts_file(out + "/counts.csv"), gaps(out + "/gaps.csv"),
        previous(out + "/predecessors.csv");
    previous << "case,window,order,cohort,scope,preceding,current,events\n";
    counts_file << "case,window,order,cohort,scope," << headers << '\n';
    gaps << "case,window,order,cohort,scope,measure,observed,negative,p10,p50,p90\n";
    Sequence full;
    full.reserve(log.size());
    for (auto &e : log)
        full.push_back(&e);
    for (bool arrival : {true, false}) {
        if (!arrival) {
            std::sort(full.begin(), full.end(),
                      [](const Event *a, const Event *b) { return a->idop < b->idop; });
            for (size_t i = 1; i < full.size(); ++i)
                if (full[i]->idop == full[i - 1]->idop)
                    throw std::runtime_error("duplicate admitted dynamic operand");
        }
        std::array<Sequence, 4> islands;
        for (auto p : full) {
            U pos = (p->idop >> 2) - start, q = (pos * 4) / roi;
            if (pos - (q * roi + 3) / 4 < 250000)
                islands[q].push_back(p);
        }
        for (bool orig : {false, true})
            for (int scope = 0; scope < 4; ++scope) {
                std::string common = std::string(arrival ? "arrival" : "program") + "," +
                                     (orig ? "nopf_miss" : "all_admitted") + "," +
                                     (scope == 0   ? "global"
                                      : scope == 1 ? "pc"
                                      : scope == 2 ? "page"
                                                   : "region256k");
                auto whole = measure(full, scope, orig);
                emit(case_id + ",full," + common, whole, counts_file, gaps, previous);
                Counts combined;
                for (unsigned q = 0; q < 4; ++q) {
                    auto small = measure(islands[q], scope, orig);
                    combined.append(small);
                    emit(case_id + ",sample_q" + std::to_string(q) + "," + common, small, counts_file, gaps,
                         previous);
                }
                emit(case_id + ",four_islands," + common, combined, counts_file, gaps, previous);
            }
    }
    return (!counts_file || !gaps || !previous) ? 5 : 0;
}
