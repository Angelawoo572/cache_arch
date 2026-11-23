/*
 * gem5's signature-path prefetcher, version 2, with a bound on one lookahead
 * walk.
 *
 * In gem5 26.0 the walk continues while the path confidence stays above a
 * threshold. Each step multiplies the confidence by
 *     (useful prefetches / issued prefetches) * (stride count / signature count).
 * When every issued prefetch so far was useful and the signature has a single
 * stride, both factors are 1, the confidence never falls, and the walk does
 * not end: the simulator stops advancing and its memory grows without bound.
 * Observed on GAP connected components, kron20, 40M instructions after the
 * marker (raw/diag/spp_hang.log).
 *
 * This class changes nothing else. A walk that would have ended on its own
 * within `max_lookahead_steps` steps runs exactly as in gem5.
 */

#ifndef __NP_EXT_NP_SPP_GUARD_HH__
#define __NP_EXT_NP_SPP_GUARD_HH__

#include <vector>

#include "base/statistics.hh"
#include "mem/cache/prefetch/signature_path_v2.hh"
#include "params/NpSignaturePathPrefetcherV2.hh"

namespace gem5
{

namespace prefetch
{

class NpSignaturePathV2 : public SignaturePathV2
{
  public:
    NpSignaturePathV2(const NpSignaturePathPrefetcherV2Params &p);

    void calculatePrefetch(const PrefetchInfo &pfi,
                           std::vector<AddrPriority> &addresses,
                           const CacheAccessor &cache) override;

  protected:
    double calculateLookaheadConfidence(
        PatternEntry const &sig,
        PatternStrideEntry const &lookahead) const override;

  private:
    const unsigned maxLookaheadSteps;
    /** Steps taken by the walk in progress. */
    mutable unsigned lookaheadSteps = 0;

    struct GuardStats : public statistics::Group
    {
        GuardStats(statistics::Group *parent);
        statistics::Scalar lookaheadsStopped;
    };
    mutable GuardStats guardStats;
};

} // namespace prefetch
} // namespace gem5

#endif // __NP_EXT_NP_SPP_GUARD_HH__
