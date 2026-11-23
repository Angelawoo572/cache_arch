/*
 * Capture streams for the gem5_neural_prefetch study.
 * See gem5_neural_prefetch/DATA_COLLECTION.md for the row formats.
 */

#include "np_ext/np_capture.hh"

#include <zlib.h>

#include <algorithm>
#include <cstring>

#include "base/logging.hh"
#include "cpu/o3/cpu.hh"
#include "cpu/o3/dyn_inst.hh"
#include "cpu/reg_class.hh"
#include "mem/cache/base.hh"
#include "mem/request.hh"
#include "sim/core.hh"
#include "sim/cur_tick.hh"

namespace gem5
{

namespace o3
{

namespace
{

/** A cache listener that also carries the cache level it is attached to. */
class CacheListener : public ProbeListenerArgBase<CacheAccessProbeArg>
{
  public:
    using Handler = void (NpCapture::*)(int, const CacheAccessProbeArg &);

    CacheListener(NpCapture *_obj, std::string _name, int _level,
                  Handler _fn)
        : ProbeListenerArgBase<CacheAccessProbeArg>(std::move(_name)),
          obj(_obj), level(_level), fn(_fn)
    {}

    void
    notify(const CacheAccessProbeArg &arg) override
    {
        (obj->*fn)(level, arg);
    }

  private:
    NpCapture *obj;
    int level;
    Handler fn;
};

inline uint64_t
regId16(const RegId &reg)
{
    return ((uint64_t(reg.classValue()) & 0xf) << 12) |
           (uint64_t(reg.index()) & 0xfff);
}

inline void
fnv1a(uint64_t &hash, uint64_t value)
{
    for (int i = 0; i < 8; ++i) {
        hash ^= (value >> (8 * i)) & 0xff;
        hash *= 1099511628211ULL;
    }
}

} // anonymous namespace

/*
 * NpStream
 */

void
NpStream::open(const std::string &dir, const std::string &_name,
               unsigned _width)
{
    name = _name;
    width = _width;
    buf.clear();
    buf.reserve(memberRows * width);
    firstRow = totalRows = byteOffset = 0;

    const std::string base = dir + "/" + name;
    data = std::fopen((base + ".u64le.gz").c_str(), "wb");
    panic_if(!data, "NpCapture cannot create %s.u64le.gz", base);
    index = std::fopen((base + ".index.csv").c_str(), "w");
    panic_if(!index, "NpCapture cannot create %s.index.csv", base);
    std::fprintf(index, "first_row,rows,byte_offset,compressed_bytes\n");
}

void
NpStream::append(const uint64_t *vals)
{
    buf.insert(buf.end(), vals, vals + width);
    ++totalRows;
    if (buf.size() >= memberRows * width)
        flushMember();
}

void
NpStream::append(std::initializer_list<uint64_t> vals)
{
    panic_if(vals.size() != width, "NpStream %s: row has %d values, not %d",
             name, vals.size(), width);
    append(vals.begin());
}

void
NpStream::flushMember()
{
    if (buf.empty())
        return;

    const uLong in_bytes = buf.size() * sizeof(uint64_t);
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    // 15 + 16 selects a gzip wrapper, so each member is a standalone gzip.
    int rc = deflateInit2(&zs, 3, Z_DEFLATED, 15 + 16, 8,
                          Z_DEFAULT_STRATEGY);
    panic_if(rc != Z_OK, "NpStream %s: deflateInit2 failed", name);

    std::vector<Bytef> out(deflateBound(&zs, in_bytes));
    zs.next_in = reinterpret_cast<Bytef *>(buf.data());
    zs.avail_in = in_bytes;
    zs.next_out = out.data();
    zs.avail_out = out.size();
    rc = deflate(&zs, Z_FINISH);
    panic_if(rc != Z_STREAM_END, "NpStream %s: deflate failed", name);
    const uint64_t out_bytes = out.size() - zs.avail_out;
    deflateEnd(&zs);

    const size_t written = std::fwrite(out.data(), 1, out_bytes, data);
    panic_if(written != out_bytes, "NpStream %s: short write", name);

    const uint64_t rows_here = buf.size() / width;
    std::fprintf(index, "%llu,%llu,%llu,%llu\n",
                 (unsigned long long)firstRow,
                 (unsigned long long)rows_here,
                 (unsigned long long)byteOffset,
                 (unsigned long long)out_bytes);
    firstRow += rows_here;
    byteOffset += out_bytes;
    buf.clear();
}

void
NpStream::close()
{
    if (!data)
        return;
    flushMember();
    std::fclose(data);
    std::fclose(index);
    data = index = nullptr;
}

/*
 * NpCapture
 */

NpCapture::NpCapture(const NpCaptureParams &params)
    : ProbeListenerObject(params),
      outDir(params.out_dir),
      contentLevel(params.content_level),
      recordUops(params.record_uops),
      startInst(params.start_inst),
      measureInsts(params.measure_insts),
      caches{nullptr, params.l1d, params.l2, params.l3}
{
    registerExitCallback([this]() { finish(); });
}

void
NpCapture::regProbeListeners()
{
    typedef ProbeListenerArg<NpCapture, DynInstPtr> InstListener;
    connectListener<InstListener>(this, "Commit", &NpCapture::onCommit);
    connectListener<InstListener>(this, "ToCommit", &NpCapture::onToCommit);

    typedef ProbeListenerArg<NpCapture, std::pair<DynInstPtr, PacketPtr>>
        DataListener;
    connectListener<DataListener>(this, "DataAccessComplete",
                                  &NpCapture::onDataAccessComplete);

    for (int level = 1; level <= 3; ++level) {
        if (!caches[level])
            continue;
        ProbeManager *pm = caches[level]->getProbeManager();
        listeners.push_back(pm->connect<CacheListener>(
            this, "Hit", level, &NpCapture::onCacheHit));
        listeners.push_back(pm->connect<CacheListener>(
            this, "Miss", level, &NpCapture::onCacheMiss));
        listeners.push_back(pm->connect<CacheListener>(
            this, "Fill", level, &NpCapture::onCacheFill));
    }
}

void
NpCapture::setEnabled(bool on)
{
    panic_if(finished, "NpCapture was already finished");
    if (on && !opened) {
        if (recordUops) {
            sUop.open(outDir, "uop", 5);
            sLoad.open(outDir, "load", 5);
            sStore.open(outDir, "store", 5);
        }
        sLoadTime.open(outDir, "load_time", 9);
        sCache.open(outDir, "cache", 6);
        sFill.open(outDir, "fill_content", 11);
        opened = true;
        enableTick = curTick();
        lastCommitTick = curTick();
    }
    enabled = on;
}

void
NpCapture::finish()
{
    if (finished || !opened)
        return;
    enabled = false;
    finished = true;
    finishTick = curTick();

    const std::string path = outDir + "/capture_counts.json";
    FILE *out = std::fopen(path.c_str(), "w");
    panic_if(!out, "NpCapture cannot create %s", path);
    std::fprintf(out, "{\n");
    std::fprintf(out, "  \"enable_tick\": %llu,\n",
                 (unsigned long long)enableTick);
    std::fprintf(out, "  \"finish_tick\": %llu,\n",
                 (unsigned long long)finishTick);
    std::fprintf(out, "  \"instructions_before_window\": %llu,\n",
                 (unsigned long long)startInst);
    std::fprintf(out, "  \"instructions_at_finish\": %llu,\n",
                 (unsigned long long)instCount);
    std::fprintf(out, "  \"committed_uops\": %llu,\n",
                 (unsigned long long)uopCount);
    std::fprintf(out, "  \"committed_loads\": %llu,\n",
                 (unsigned long long)loadCount);
    std::fprintf(out, "  \"written_back_stores\": %llu,\n",
                 (unsigned long long)storeCount);
    std::fprintf(out, "  \"stores_pending_at_finish\": %llu,\n",
                 (unsigned long long)pendingStores.size());
    std::fprintf(out, "  \"v1_load_hash\": \"%016llx\",\n",
                 (unsigned long long)v1Hash);
    std::fprintf(out, "  \"rows\": {");
    bool first = true;
    for (NpStream *s : {&sUop, &sLoad, &sStore, &sLoadTime, &sCache,
                        &sFill}) {
        if (!s->isOpen())
            continue;
        std::fprintf(out, "%s\"%s\": %llu", first ? "" : ", ",
                     s->getName().c_str(), (unsigned long long)s->rows());
        first = false;
    }
    std::fprintf(out, "}\n}\n");
    std::fclose(out);

    for (NpStream *s : {&sUop, &sLoad, &sStore, &sLoadTime, &sCache,
                        &sFill})
        s->close();
}

void
NpCapture::onCommit(const DynInstPtr &inst)
{
    const Tick now = curTick();
    const InstSeqNum seq = inst->seqNum;
    const bool first_uop = !inst->isMicroop() || inst->isFirstMicroop();
    const bool last_uop = !inst->isMicroop() || inst->isLastMicroop();

    // Open the window on an instruction boundary, at an exact count.
    if (measureInsts && !opened && !finished && first_uop &&
        instCount >= startInst) {
        setEnabled(true);
    }

    if (enabled) {
        const PCStateBase &pc = inst->pcState();
        const uint64_t pc_addr = pc.instAddr();
        const bool is_load = inst->isLoad();
        const bool is_store = inst->isStore();

        if (recordUops) {
            const size_t nsrc = inst->numSrcRegs();
            const size_t ndst = inst->numDestRegs();

            uint64_t meta = (uint64_t(pc.microPC()) & 0xff) |
                            ((uint64_t(inst->opClass()) & 0xff) << 8);
            if (is_load) meta |= 1ULL << 16;
            if (is_store) meta |= 1ULL << 17;
            if (inst->isControl()) meta |= 1ULL << 18;
            if (inst->isIndirectCtrl()) meta |= 1ULL << 19;
            if (inst->isCall()) meta |= 1ULL << 20;
            if (inst->isReturn()) meta |= 1ULL << 21;
            if (inst->isCondCtrl()) meta |= 1ULL << 22;
            meta |= uint64_t(std::min<size_t>(nsrc, 255)) << 32;
            meta |= uint64_t(std::min<size_t>(ndst, 255)) << 40;

            // The first four integer-class sources.
            uint64_t src = ~0ULL;
            unsigned k = 0;
            for (size_t i = 0; i < nsrc && k < 4; ++i) {
                const RegId &reg = inst->srcRegIdx(i);
                if (!reg.is(IntRegClass))
                    continue;
                src &= ~(0xffffULL << (16 * k));
                src |= regId16(reg) << (16 * k);
                ++k;
            }

            // The first two integer-class destinations, and the value
            // written to the first of them.
            uint64_t dst = ~0ULL;
            uint64_t dst_value = 0;
            k = 0;
            for (size_t i = 0; i < ndst && k < 2; ++i) {
                const RegId &reg = inst->destRegIdx(i);
                if (!reg.is(IntRegClass))
                    continue;
                dst &= ~(0xffffULL << (16 * k));
                dst |= regId16(reg) << (16 * k);
                if (k == 0) {
                    // The commit rename map already points at this
                    // micro-op's result. This read updates no statistic.
                    dst_value =
                        inst->cpu->getArchReg(reg, inst->threadNumber);
                    meta |= 1ULL << 23;
                }
                ++k;
            }

            sUop.append({pc_addr, meta, src, dst, dst_value});
        }

        if (is_load) {
            const unsigned size = inst->effSize;
            uint64_t size_flags = size & 0xff;
            if (((inst->effAddr & 63) + size) > 64)
                size_flags |= 1ULL << 8;
            if (size > 8)
                size_flags |= 1ULL << 9;
            uint64_t value = 0;
            if (inst->memData)
                std::memcpy(&value, inst->memData,
                            std::min<unsigned>(size, 8));
            else
                size_flags |= 1ULL << 10;

            if (recordUops) {
                sLoad.append({uopCount, pc_addr, inst->effAddr, size_flags,
                              value});
            }
            fnv1a(v1Hash, pc_addr);
            fnv1a(v1Hash, inst->effAddr);
            fnv1a(v1Hash, size_flags);
            fnv1a(v1Hash, value);

            Tick t_data = 0;
            uint64_t service = 0;
            auto it = loadData.find(seq);
            if (it != loadData.end()) {
                t_data = it->second.tick;
                service = it->second.viaCache ? 1 : 0;
            }
            const Tick t_rename = inst->renameTick >= 0 ?
                inst->fetchTick + inst->renameTick : 0;
            sLoadTime.append({seq, inst->physEffAddr, inst->fetchTick,
                              t_rename, inst->firstIssue, t_data,
                              lastCommitTick, now, service});
            ++loadCount;
        }

        if (is_store && recordUops) {
            const unsigned size = inst->effSize;
            uint64_t size_flags = size & 0xff;
            if (((inst->effAddr & 63) + size) > 64)
                size_flags |= 1ULL << 8;
            if (size > 8)
                size_flags |= 1ULL << 9;
            pendingStores[seq] =
                PendingStore{uopCount, pc_addr, inst->effAddr, size_flags};
            // A store that never writes back must not pin memory forever.
            if (pendingStores.size() > 4096)
                pendingStores.erase(pendingStores.begin());
        }

        ++uopCount;
    }

    // Everything at or before this sequence number has left the pipeline.
    loadData.erase(loadData.begin(), loadData.upper_bound(seq));
    lastCommitTick = now;

    if (last_uop) {
        ++instCount;
        if (measureInsts && enabled &&
            instCount >= startInst + measureInsts) {
            finish();
        }
    }
}

void
NpCapture::onToCommit(const DynInstPtr &inst)
{
    // A load forwarded from a store never fires DataAccessComplete. Its
    // data is in hand when it is sent to commit.
    if (inst->isLoad())
        loadData.emplace(inst->seqNum, DataInfo{curTick(), false});
}

void
NpCapture::onDataAccessComplete(
    const std::pair<DynInstPtr, PacketPtr> &arg)
{
    const DynInstPtr &inst = arg.first;
    if (inst->isLoad()) {
        loadData[inst->seqNum] = DataInfo{curTick(), true};
        return;
    }

    if (!enabled || !recordUops || !inst->isStore())
        return;
    auto it = pendingStores.find(inst->seqNum);
    if (it == pendingStores.end())
        return;

    uint64_t value = 0;
    const unsigned size = it->second.sizeFlags & 0xff;
    if (inst->memData) {
        std::memcpy(&value, inst->memData, std::min<unsigned>(size, 8));
    } else if (arg.second && arg.second->getSize() > 0) {
        std::memcpy(&value, arg.second->getConstPtr<uint8_t>(),
                    std::min<unsigned>(arg.second->getSize(), 8));
    }
    sStore.append({it->second.uop, it->second.pc, it->second.vaddr,
                   it->second.sizeFlags, value});
    ++storeCount;
    pendingStores.erase(it);
}

void
NpCapture::onCacheHit(int level, const CacheAccessProbeArg &arg)
{
    cacheEvent(level, EvHit, arg);
}

void
NpCapture::onCacheMiss(int level, const CacheAccessProbeArg &arg)
{
    cacheEvent(level, EvMiss, arg);
}

void
NpCapture::onCacheFill(int level, const CacheAccessProbeArg &arg)
{
    cacheEvent(level, EvFill, arg);
}

void
NpCapture::cacheEvent(int level, unsigned event,
                      const CacheAccessProbeArg &arg)
{
    if (!enabled)
        return;

    const PacketPtr pkt = arg.pkt;
    const RequestPtr &req = pkt->req;
    const Addr addr = pkt->getAddr();

    uint64_t code = uint64_t(level) | (uint64_t(event) << 4);
    if (event == EvHit &&
        arg.cache.hasBeenPrefetched(addr, pkt->isSecure()))
        code |= 1ULL << 8;
    if (pkt->cmd.isHWPrefetch())
        code |= 1ULL << 9;
    if (pkt->isWrite() || pkt->needsWritable())
        code |= 1ULL << 10;
    if (req->isInstFetch())
        code |= 1ULL << 11;
    if (pkt->cmd.isSWPrefetch())
        code |= 1ULL << 12;
    code |= (uint64_t(req->requestorId()) & 0xffff) << 16;

    // An ordinary L1D hit says nothing a missing S6 row does not already
    // say. Keep only hits that are the first use of a prefetched line or
    // that carry a prefetch.
    const uint64_t keep = (1ULL << 8) | (1ULL << 9) | (1ULL << 12);
    if (event == EvHit && level == 1 && !(code & keep))
        return;

    const uint64_t vline = req->hasVaddr() ? (req->getVaddr() >> 6) : 0;
    const uint64_t who =
        req->hasInstSeqNum() ? req->getReqInstSeqNum() : 0;
    const uint64_t pc = req->hasPC() ? req->getPC() : 0;
    sCache.append({curTick(), code, addr >> 6, vline, who, pc});

    if (event == EvFill && unsigned(level) == contentLevel &&
        !req->isInstFetch() && pkt->hasData() && pkt->getSize() == 64) {
        uint64_t row[11];
        row[0] = curTick();
        row[1] = addr >> 6;
        row[2] = vline;
        std::memcpy(&row[3], pkt->getConstPtr<uint8_t>(), 64);
        sFill.append(row);
    }
}

} // namespace o3
} // namespace gem5
