from m5.objects.Prefetcher import SignaturePathPrefetcherV2
from m5.params import *


class NpSignaturePathPrefetcherV2(SignaturePathPrefetcherV2):
    """gem5's SignaturePathPrefetcherV2 with a bound on one lookahead walk.

    Without the bound a walk does not end when its confidence stops falling
    (see np_spp_guard.hh). A walk that ends on its own within the bound is
    unchanged.
    """

    type = "NpSignaturePathPrefetcherV2"
    cxx_class = "gem5::prefetch::NpSignaturePathV2"
    cxx_header = "np_ext/np_spp_guard.hh"

    max_lookahead_steps = Param.Unsigned(
        1024, "Steps after which one lookahead walk is stopped"
    )
