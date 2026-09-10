#!/usr/bin/env python3
"""Exercise the original ListReplayer with a software cache fixture; no NN changes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_measurement import simulator_source


class OriginalReplayTests(unittest.TestCase):
    def test_original_warmup_and_interleaved_occurrence_keys(self):
        source = simulator_source()
        with tempfile.TemporaryDirectory(prefix="dynamic602_replay_") as tmp:
            stage = Path(tmp)
            for part in ("inc/list_replayer.h", "inc/prefetcher.h", "prefetcher/list_replayer.cc"):
                shutil.copy2(source / part, stage / Path(part).name)
            (stage / "cache.h").write_text("#pragma once\nclass CACHE {public: unsigned cpu=0;};\n")
            (stage / "champsim.h").write_text("#pragma once\n#include <cstdint>\nconst unsigned LOAD=0, BLOCK_SIZE=64, LOG2_BLOCK_SIZE=6;\nextern unsigned char warmup_complete[1];\n")
            (stage / "actions.csv").write_text("pc,line,occ,prefetch_addr\n7,100,0,6400\n7,100,0,6464\n9,200,0,12864\n7,100,1,6592\n")
            (stage / "fixture.cc").write_text(r'''
#include <cstdint>
#include "list_replayer.h"
#include "cache.h"
#include "champsim.h"
#include <cassert>
#include <iostream>
unsigned char warmup_complete[1]={0};
int main(){
  CACHE cache; ListReplayer replay("list_replayer",&cache);
  std::vector<uint64_t> output;
  replay.invoke_prefetcher(7,6400,0,LOAD,output);assert(output.empty());
  warmup_complete[0]=1;
  replay.invoke_prefetcher(7,6400,0,1,output);assert(output.empty());
  replay.invoke_prefetcher(7,6400,0,LOAD,output);assert((output==std::vector<uint64_t>{6400,6464}));output.clear();
  replay.invoke_prefetcher(9,12800,0,LOAD,output);assert((output==std::vector<uint64_t>{12864}));output.clear();
  replay.invoke_prefetcher(7,6400,0,LOAD,output);assert((output==std::vector<uint64_t>{6592}));output.clear();
  replay.invoke_prefetcher(99,6400,0,LOAD,output);assert(output.empty());
  replay.invoke_prefetcher(7,6400,0,LOAD,output);assert(output.empty());
  replay.dump_stats();std::cout<<"PASS original replay post-warmup LOAD domain and PC-line-occ keys\n";
}
''')
            compiler = os.environ.get("CXX") or shutil.which("c++") or "g++"
            subprocess.run([compiler,"-std=c++11","-O2","-I"+str(stage),str(stage/"fixture.cc"),str(stage/"list_replayer.cc"),"-o",str(stage/"fixture")],check=True,capture_output=True,text=True)
            result = subprocess.run([str(stage/"fixture")],env=dict(os.environ,PFETCH_LIST_PATH=str(stage/"actions.csv")),check=True,capture_output=True,text=True)
            self.assertIn("PASS",result.stdout)
            self.assertIn("emitted 4 candidates over 5 runtime ROI L2 LOAD accesses (3 matched PC-line-occ triggers",result.stderr)
            print(result.stdout.strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
