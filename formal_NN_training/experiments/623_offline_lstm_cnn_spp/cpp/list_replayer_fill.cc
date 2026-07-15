#include "list_replayer_fill.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "cache.h"
#include "champsim.h"

using namespace std;

namespace {
string trim_copy(string value)
{
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), find_if(value.begin(), value.end(), not_space));
    value.erase(find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

bool split_exact(const string& line, size_t expected, vector<string>& fields)
{
    fields.clear();
    string item;
    stringstream stream(line);
    while (getline(stream, item, ',')) fields.push_back(trim_copy(item));
    return fields.size() == expected;
}
} // namespace

ListReplayerFill::ListReplayerFill(string type, CACHE* cache)
    : Prefetcher(type), cache_(cache)
{
    load_table();
}

ListReplayerFill::~ListReplayerFill() = default;

void ListReplayerFill::load_table()
{
    const char* env = std::getenv("PFETCH_LIST_PATH");
    path_ = (env && env[0]) ? env : "/tmp/prefetch_list_fill.csv";
    ifstream input(path_);
    if (!input) {
        throw runtime_error("[list_replayer_fill] could not open " + path_);
    }

    string raw;
    uint64_t line_number = 0;
    vector<string> fields;
    bool saw_header = false;
    while (getline(input, raw)) {
        ++line_number;
        const string line = trim_copy(raw);
        if (line.empty() || line[0] == '#') continue;
        if (!saw_header) {
            if (line != "pc,line,occ,prefetch_addr,fill_level") {
                throw runtime_error(
                    "[list_replayer_fill] missing five-column header in " + path_
                );
            }
            saw_header = true;
            continue;
        }
        if (!split_exact(line, 5, fields) || fields[0].empty()) {
            throw runtime_error(
                "[list_replayer_fill] malformed row " + to_string(line_number)
                + " in " + path_
            );
        }
        try {
            TriggerKey key;
            key.pc = stoull(fields[0], nullptr, 0);
            key.line = stoull(fields[1], nullptr, 0);
            key.occ = stoull(fields[2], nullptr, 10);
            Action action;
            action.address = stoull(fields[3], nullptr, 0);
            action.fill_level = stoi(fields[4], nullptr, 0);
            if (
                (action.address & (BLOCK_SIZE - 1)) != 0
                || (action.fill_level != FILL_L2 && action.fill_level != FILL_LLC)
            ) {
                throw runtime_error("invalid aligned address/fill level");
            }
            table_[key].push_back(action);
            ++loaded_;
        } catch (const exception& error) {
            throw runtime_error(
                "[list_replayer_fill] invalid row " + to_string(line_number)
                + " in " + path_ + ": " + error.what()
            );
        }
    }
    if (!saw_header) {
        throw runtime_error(
            "[list_replayer_fill] replay list has no valid header " + path_
        );
    }
    cerr << "[list_replayer_fill] loaded " << loaded_ << " direct actions across "
         << table_.size() << " PC-line-occ triggers from " << path_ << endl;
}

void ListReplayerFill::invoke_prefetcher(
    uint64_t pc, uint64_t address, uint8_t /*cache_hit*/, uint8_t type,
    vector<uint64_t>& /*pref_addr*/)
{
    if (type != LOAD || !warmup_complete[cache_->cpu]) return;

    const uint64_t line = address >> LOG2_BLOCK_SIZE;
    PairKey pair;
    pair.pc = pc;
    pair.line = line;
    TriggerKey key;
    key.pc = pc;
    key.line = line;
    key.occ = occurrences_[pair]++;

    const auto found = table_.find(key);
    if (found != table_.end()) {
        ++matched_;
        for (const Action& action : found->second) {
            cache_->prefetch_line(pc, address, action.address, action.fill_level, 0);
            ++emitted_;
            if (action.fill_level == FILL_L2) ++emitted_l2_;
            else ++emitted_llc_;
        }
    }
    ++callbacks_;
}

void ListReplayerFill::dump_stats()
{
    if (stats_dumped_) return;
    stats_dumped_ = true;
    cerr << "[list_replayer_fill] emitted " << emitted_ << " actions over "
         << callbacks_ << " runtime ROI L2 LOAD accesses (" << matched_
         << " matched PC-line-occ triggers; " << table_.size()
         << " loaded trigger keys; key=pc_line_occ; fill_l2=" << emitted_l2_
         << "; fill_llc=" << emitted_llc_ << ")" << endl;
}

void ListReplayerFill::print_config()
{
    cout << "list_replayer_input " << path_ << endl
         << "list_replayer_trigger_key pc_line_occ" << endl
         << "list_replayer_action_metadata captured_fill_level" << endl
         << "list_replayer_domain ROI_L2_LOAD_after_warmup" << endl;
}
