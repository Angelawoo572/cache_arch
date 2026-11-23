from m5.objects.Probe import ProbeListenerObject
from m5.params import *
from m5.proxy import *
from m5.util.pybind import PyBindMethod


class NpCapture(ProbeListenerObject):
    """Records the capture streams of gem5_neural_prefetch/DATA_COLLECTION.md.

    `manager` must be the O3 CPU. Listeners only: no simulated state is touched.
    """

    type = "NpCapture"
    cxx_class = "gem5::o3::NpCapture"
    cxx_header = "np_ext/np_capture.hh"

    cxx_exports = [
        PyBindMethod("setEnabled"),
        PyBindMethod("finish"),
    ]

    out_dir = Param.String("Directory that receives the capture streams")
    l1d = Param.BaseCache(NULL, "L1 data cache to observe")
    l2 = Param.BaseCache(NULL, "L2 cache to observe")
    l3 = Param.BaseCache(NULL, "L3 cache to observe")
    content_level = Param.Unsigned(
        2, "Cache level whose fills are recorded with line contents (S8)"
    )
    record_uops = Param.Bool(
        True, "Record the architectural micro-op, load and store streams"
    )
    start_inst = Param.Counter(
        0,
        "Start recording at the first micro-op after this many committed "
        "instructions. With measure_insts = 0 the window is manual.",
    )
    measure_insts = Param.Counter(
        0, "Stop after exactly this many recorded instructions (0 = manual)"
    )
