/*
 * Capture streams for the gem5_neural_prefetch study.
 * See gem5_neural_prefetch/DATA_COLLECTION.md for the row formats.
 */

#ifndef __NP_EXT_NP_CAPTURE_HH__
#define __NP_EXT_NP_CAPTURE_HH__

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "base/types.hh"
#include "cpu/inst_seq.hh"
#include "cpu/o3/dyn_inst_ptr.hh"
#include "mem/cache/cache_probe_arg.hh"
#include "mem/packet.hh"
#include "params/NpCapture.hh"
#include "sim/probe/probe_listener_object.hh"

namespace gem5
{

class BaseCache;

namespace o3
{

/**
 * A table of fixed-width rows of little-endian unsigned 64-bit integers,
 * written as independent gzip members of 32,768 rows with a CSV index.
 */
class NpStream
{
  public:
    static constexpr uint64_t memberRows = 32768;

    void open(const std::string &dir, const std::string &name,
              unsigned width);
    void append(const uint64_t *vals);
    void append(std::initializer_list<uint64_t> vals);
    void close();
    uint64_t rows() const { return totalRows; }
    bool isOpen() const { return data != nullptr; }
    const std::string &getName() const { return name; }

  private:
    void flushMember();

    std::string name;
    unsigned width = 0;
    std::vector<uint64_t> buf;
    FILE *data = nullptr;
    FILE *index = nullptr;
    uint64_t firstRow = 0;
    uint64_t totalRows = 0;
    uint64_t byteOffset = 0;
};

class NpCapture : public ProbeListenerObject
{
  public:
    NpCapture(const NpCaptureParams &params);

    void regProbeListeners() override;

    /** Start or pause recording. Counters restart on the first enable. */
    void setEnabled(bool on);

    /** Flush and close every stream and write capture_counts.json. */
    void finish();

  private:
    enum CacheEvent : unsigned { EvHit = 1, EvMiss = 2, EvFill = 4 };

    void onCommit(const DynInstPtr &inst);
    void onToCommit(const DynInstPtr &inst);
    void onDataAccessComplete(const std::pair<DynInstPtr, PacketPtr> &arg);

    void onCacheHit(int level, const CacheAccessProbeArg &arg);
    void onCacheMiss(int level, const CacheAccessProbeArg &arg);
    void onCacheFill(int level, const CacheAccessProbeArg &arg);
    void cacheEvent(int level, unsigned event,
                    const CacheAccessProbeArg &arg);

    const std::string outDir;
    const unsigned contentLevel;
    const bool recordUops;
    /** Committed instructions to skip before the window, and its length.
     *  The window is counted here, at commit, so that every arm records
     *  exactly the same instructions whatever its timing. */
    const uint64_t startInst;
    const uint64_t measureInsts;
    BaseCache *caches[4];

    bool opened = false;
    bool enabled = false;
    bool finished = false;

    /** Whole instructions committed since the start of simulation. */
    uint64_t instCount = 0;
    Tick finishTick = 0;
    uint64_t uopCount = 0;
    uint64_t loadCount = 0;
    uint64_t storeCount = 0;
    uint64_t v1Hash = 14695981039346656037ULL;
    uint64_t valueHash = 14695981039346656037ULL;
    Tick enableTick = 0;
    Tick lastCommitTick = 0;

    struct DataInfo
    {
        Tick tick;
        bool viaCache;
    };
    /** When each in-flight load's data returned. */
    std::map<InstSeqNum, DataInfo> loadData;

    struct PendingStore
    {
        uint64_t uop;
        uint64_t pc;
        uint64_t vaddr;
        uint64_t sizeFlags;
    };
    /** Committed stores that have not written back yet. */
    std::map<InstSeqNum, PendingStore> pendingStores;

    NpStream sUop;
    NpStream sLoad;
    NpStream sStore;
    NpStream sLoadTime;
    NpStream sCache;
    NpStream sFill;
};

} // namespace o3
} // namespace gem5

#endif // __NP_EXT_NP_CAPTURE_HH__
