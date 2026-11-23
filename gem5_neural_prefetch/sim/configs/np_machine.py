"""The simulated machine shared by the checkpoint and run scripts.

One x86-64 core in syscall-emulation mode, three levels of classic cache and
one DDR4 channel. An arm chooses which conventional prefetchers are present.
"""

from m5.objects import (
    AMPMPrefetcher,
    BadAddr,
    BOPPrefetcher,
    Cache,
    IndirectMemoryPrefetcher,
    L2XBar,
    NULL,
    SignaturePathPrefetcherV2,
    StridePrefetcher,
    SystemXBar,
)

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.cachehierarchies.classic.abstract_classic_cache_hierarchy import (
    AbstractClassicCacheHierarchy,
)
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.memory import SingleChannelDDR4_2400
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.isas import ISA

CLOCK = "4GHz"
MEMORY = "8GiB"

# The workloads execute the gem5 work-begin marker only when this is set, so the
# same binaries also run natively.
WORKLOAD_ENV = ["NP_GEM5=1"]

# Arm name -> prefetcher at L1D and at L2. The LLC has none in any arm.
ARMS = {
    "nopf": {"l1d": None, "l2": None},
    "stride": {"l1d": "stride", "l2": None},
    "fast": {"l1d": "stride", "l2": "spp"},
    "bop": {"l1d": "stride", "l2": "bop"},
    "ampm": {"l1d": "stride", "l2": "ampm"},
    "imp": {"l1d": "imp", "l2": None},
}

# Core sizes. Everything not listed keeps gem5's O3 default.
O3_SIZES = {
    "numROBEntries": 256,
    "LQEntries": 96,
    "SQEntries": 64,
    "numPhysIntRegs": 320,
    "numPhysFloatRegs": 320,
}

# size, associativity, tag, data and response latency in cycles, MSHRs,
# targets per MSHR
CACHES = {
    "l1i": ("32KiB", 8, 1, 1, 1, 8, 20),
    "l1d": ("48KiB", 12, 4, 4, 1, 16, 20),
    "l2": ("512KiB", 8, 10, 10, 2, 32, 20),
    "l3": ("2MiB", 16, 20, 20, 2, 64, 20),
}


def make_prefetcher(kind):
    if kind is None:
        return NULL
    if kind == "stride":
        return StridePrefetcher()
    if kind == "spp":
        return SignaturePathPrefetcherV2()
    if kind == "bop":
        return BOPPrefetcher()
    if kind == "ampm":
        return AMPMPrefetcher()
    if kind == "imp":
        # The indirect-memory prefetcher reads index values from hits, which
        # a prefetcher only sees with these two settings.
        return IndirectMemoryPrefetcher(
            prefetch_on_access=True, prefetch_on_pf_hit=False
        )
    raise ValueError(f"unknown prefetcher {kind!r}")


def make_cache(level, prefetcher=NULL):
    size, assoc, tag, data, response, mshrs, targets = CACHES[level]
    cache = Cache(
        size=size,
        assoc=assoc,
        tag_latency=tag,
        data_latency=data,
        response_latency=response,
        mshrs=mshrs,
        tgts_per_mshr=targets,
    )
    cache.prefetcher = prefetcher
    return cache


class ThreeLevelHierarchy(AbstractClassicCacheHierarchy):
    """Private L1I, L1D and L2 over one L3, for a single core."""

    def __init__(self, arm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choose from {list(ARMS)}")
        choice = ARMS[arm]

        self.membus = SystemXBar(width=64)
        self.membus.badaddr_responder = BadAddr()
        self.membus.default = self.membus.badaddr_responder.pio

        # The caches exist from construction so that a capture object can
        # refer to them before the board wires them up.
        self.l1i = make_cache("l1i")
        self.l1d = make_cache("l1d", make_prefetcher(choice["l1d"]))
        self.l2 = make_cache("l2", make_prefetcher(choice["l2"]))
        self.l3 = make_cache("l3")
        self.l2bus = L2XBar()
        self.l3bus = L2XBar()

    def get_mem_side_port(self):
        return self.membus.mem_side_ports

    def get_cpu_side_port(self):
        return self.membus.cpu_side_ports

    def incorporate_cache(self, board):
        board.connect_system_port(self.membus.cpu_side_ports)
        for _, port in board.get_mem_ports():
            self.membus.mem_side_ports = port

        cores = board.get_processor().get_cores()
        if len(cores) != 1:
            raise ValueError("this hierarchy is written for one core")
        cpu = cores[0]

        self.l1i.mem_side = self.l2bus.cpu_side_ports
        self.l1d.mem_side = self.l2bus.cpu_side_ports
        self.l2bus.mem_side_ports = self.l2.cpu_side
        self.l2.mem_side = self.l3bus.cpu_side_ports
        self.l3bus.mem_side_ports = self.l3.cpu_side
        self.l3.mem_side = self.membus.cpu_side_ports

        cpu.connect_icache(self.l1i.cpu_side)
        cpu.connect_dcache(self.l1d.cpu_side)
        cpu.connect_walker_ports(
            self.l2bus.cpu_side_ports, self.l2bus.cpu_side_ports
        )
        # x86 needs its interrupt controller on the memory bus.
        cpu.connect_interrupt(
            self.membus.mem_side_ports, self.membus.cpu_side_ports
        )


def build_board(cpu_type, hierarchy):
    """A one-core syscall-emulation board. The memory system is the same for
    every CPU type so that a checkpoint restores into any of them."""
    processor = SimpleProcessor(
        cpu_type=cpu_type, isa=ISA.X86, num_cores=1, clk_freq=CLOCK
    )
    board = SimpleBoard(
        clk_freq=CLOCK,
        processor=processor,
        memory=SingleChannelDDR4_2400(size=MEMORY),
        cache_hierarchy=hierarchy,
    )
    return board


def checkpoint_board():
    return build_board(CPUTypes.ATOMIC, NoCache())


def measured_board(arm):
    hierarchy = ThreeLevelHierarchy(arm)
    board = build_board(CPUTypes.O3, hierarchy)
    cpu = board.get_processor().get_cores()[0].get_simobject()
    for name, value in O3_SIZES.items():
        setattr(cpu, name, value)
    return board, hierarchy, cpu
