#!/usr/bin/env python3
"""Add the gem5 work-begin marker to the GAP benchmark driver.

Edits src/benchmark.h of a GAP checkout in place and is safe to run twice.
With NP_GEM5 set in the environment the patched driver
  - executes the gem5 work-begin operation before the first trial, and
  - prints no per-trial time, so that nothing printed inside the measured
    region depends on simulated time.
Without NP_GEM5 it behaves exactly as upstream.
"""

import sys
from pathlib import Path

MARK = "gem5_neural_prefetch marker"

HELPER = f"""
// {MARK}: bytes 0F 04 and the 16-bit function number 0x5a are gem5's
// work-begin operation. It is an invalid opcode on real hardware, so it is
// executed only when NP_GEM5 is set.
static void __attribute__((noinline)) NpGem5WorkBegin() {{
  __asm__ volatile(".byte 0x0F, 0x04\\n\\t.word 0x5a"
                   : : "D"(0UL), "S"(0UL) : "rax", "memory");
}}

"""

EDITS = [
    (
        "  g.PrintStats();\n  double total_seconds = 0;\n",
        "  g.PrintStats();\n"
        '  const bool np_gem5 = std::getenv("NP_GEM5") != nullptr;\n'
        "  double total_seconds = 0;\n",
    ),
    (
        "    trial_timer.Start();\n    auto result = kernel(g);\n"
        "    trial_timer.Stop();\n"
        '    PrintTime("Trial Time", trial_timer.Seconds());\n',
        "    if (np_gem5 && iter == 0)\n      NpGem5WorkBegin();\n"
        "    trial_timer.Start();\n    auto result = kernel(g);\n"
        "    trial_timer.Stop();\n"
        "    if (!np_gem5)\n"
        '      PrintTime("Trial Time", trial_timer.Seconds());\n',
    ),
]


def main():
    path = Path(sys.argv[1]) / "src" / "benchmark.h"
    text = path.read_text()
    if MARK in text:
        print(f"{path}: already patched")
        return 0
    for old, new in EDITS:
        if text.count(old) != 1:
            sys.exit(f"{path}: expected text not found exactly once:\n{old}")
        text = text.replace(old, new)
    anchor = "template<typename GraphT_, typename GraphFunc, typename AnalysisFunc,"
    if text.count(anchor) != 1:
        sys.exit(f"{path}: cannot find the BenchmarkKernel template header")
    text = text.replace(anchor, HELPER.lstrip("\n") + anchor)
    if "#include <cstdlib>" not in text:
        first_include = text.index("#include")
        text = text[:first_include] + "#include <cstdlib>\n" + text[first_include:]
    path.write_text(text)
    print(f"{path}: patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
