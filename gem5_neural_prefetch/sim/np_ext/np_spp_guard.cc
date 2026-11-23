#include "np_ext/np_spp_guard.hh"

namespace gem5
{

namespace prefetch
{

NpSignaturePathV2::NpSignaturePathV2(
    const NpSignaturePathPrefetcherV2Params &p)
    : SignaturePathV2(p),
      maxLookaheadSteps(p.max_lookahead_steps),
      guardStats(this)
{
}

NpSignaturePathV2::GuardStats::GuardStats(statistics::Group *parent)
    : statistics::Group(parent),
      ADD_STAT(lookaheadsStopped, statistics::units::Count::get(),
               "Lookahead walks stopped at the step limit")
{
}

void
NpSignaturePathV2::calculatePrefetch(const PrefetchInfo &pfi,
                                     std::vector<AddrPriority> &addresses,
                                     const CacheAccessor &cache)
{
    lookaheadSteps = 0;
    SignaturePathV2::calculatePrefetch(pfi, addresses, cache);
}

double
NpSignaturePathV2::calculateLookaheadConfidence(
    PatternEntry const &sig, PatternStrideEntry const &lookahead) const
{
    if (++lookaheadSteps > maxLookaheadSteps) {
        // A confidence of zero ends the walk in SignaturePath.
        guardStats.lookaheadsStopped++;
        return 0.0;
    }
    // The expression of SignaturePathV2::calculateLookaheadConfidence, which
    // is private there, written out unchanged.
    if (sig.counter == 0)
        return 0.0;
    return (((double) usefulPrefetches) / issuedPrefetches) *
           (((double) lookahead.counter) / sig.counter);
}

} // namespace prefetch
} // namespace gem5
