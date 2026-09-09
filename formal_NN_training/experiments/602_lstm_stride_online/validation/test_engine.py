#!/usr/bin/env python3
"""Compile/test the actual online engine against a small software cache fixture."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
sys.path.insert(0, str(EXP / "runtime"))
sys.path.insert(0, str(EXP / "python"))
from prepare_forward import prepare
from online_worker import OnlineTrainer
from formal_NN_training.common.stride_direct_action_model import FrozenStrideLiveModel, load_checkpoint
from live_model_format import write_model

STUBS = {
    "prefetcher.h": r'''#pragma once
#include <string>
class Prefetcher { public: explicit Prefetcher(std::string) {} virtual ~Prefetcher() {} };
''',
    "champsim.h": r'''#pragma once
#include <cstdint>
const unsigned char LOAD=0;
const unsigned LOG2_BLOCK_SIZE=6,FILL_L2=2;
extern unsigned char warmup_complete[1];
extern uint64_t current_core_cycle[1];
''',
    "cache.h": r'''#pragma once
#include <cstdint>
#include <vector>
class CACHE { public:
  unsigned cpu=0;
  struct Request { uint64_t pc,base,address; };
  std::vector<Request> requests;
  int prefetch_line(uint64_t pc,uint64_t base,uint64_t address,unsigned,unsigned) {
    requests.push_back(Request{pc,base,address});return 1;
  }
};
''',
    "stride.h": r'''#pragma once
#include "prefetcher.h"
#include "champsim.h"
#include <algorithm>
#include <cstdint>
#include <map>
#include <vector>
// Software label fixture only. Production compiles the existing Stride source.
class StridePrefetcher : public Prefetcher {
  struct Entry { uint64_t line,lru; int64_t stride; };
  std::map<uint64_t,Entry> entries;uint64_t tick=0;
 public:
  explicit StridePrefetcher(std::string s):Prefetcher(s){}
  void invoke_prefetcher(uint64_t pc,uint64_t address,uint8_t,uint8_t,std::vector<uint64_t>& output) {
    uint64_t line=address>>6;auto it=entries.find(pc);
    if(it==entries.end()) {
      if(entries.size()==64){auto victim=std::min_element(entries.begin(),entries.end(),[](const std::pair<const uint64_t,Entry>& a,const std::pair<const uint64_t,Entry>& b){return a.second.lru<b.second.lru;});entries.erase(victim);}
      entries[pc]=Entry{line,++tick,0};return;
    }
    int64_t stride=static_cast<int64_t>(line)-static_cast<int64_t>(it->second.line);
    if(stride==0)return;
    if(stride==it->second.stride)for(int degree=0;degree<2;++degree){int64_t offset=static_cast<int64_t>(line%64)+degree*stride;if(offset<0||offset>=64)break;output.push_back(((line/64)*64+offset)<<6);}
    it->second=Entry{line,++tick,stride};
  }
};
''',
}

PROBE = r'''
#include "prefetcher.h"
#include "stride_lstm_runtime.h"
#include "stride.h"
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>
#define private public
#include "online602_engine.h"
#undef private
#include "cache.h"
#include "champsim.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
unsigned char warmup_complete[1]={1};
uint64_t current_core_cycle[1]={0};
template<class T>void array(const std::vector<T>& values){std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
void result(const stride_lstm::InferenceResult& r){std::cout<<"{\"pc\":"<<r.pc<<",\"line\":"<<r.line<<",\"emit\":"<<r.emit<<",\"k\":"<<r.k<<",\"deltas\":";array(r.deltas);std::cout<<",\"addresses\":";array(r.addresses);std::cout<<",\"hidden\":";array(r.hidden);std::cout<<",\"cell\":";array(r.cell);std::cout<<'}';}
void snapshot(Online602& engine,CACHE& cache){
  std::cout<<"{\"stats\":{"<<engine.json()<<"},\"cycle\":"<<current_core_cycle[0]<<",\"active_due\":"<<engine.due_<<",\"train_due\":"<<engine.train_due_<<",\"publish_due\":"<<engine.publish_due_<<",\"states\":[";
  bool first=true;for(const auto& item:engine.entries_){if(!first)std::cout<<',';first=false;std::cout<<"{\"pc\":"<<item.first<<",\"life\":"<<item.second.life<<",\"hidden\":";array(item.second.state.hidden);std::cout<<",\"cell\":";array(item.second.state.cell);std::cout<<'}';}
  std::cout<<"],\"issued\":[";first=true;for(const auto& item:cache.requests){if(!first)std::cout<<',';first=false;std::cout<<'['<<item.pc<<','<<item.base<<','<<item.address<<']';}
  std::cout<<"],\"active_result\":";result(engine.result_);std::cout<<"}"<<std::endl;
}
int main(int argc,char** argv){try{
  std::cout<<std::setprecision(9);
  if(argc>1&&std::string(argv[1])=="forward"){
    stride_lstm::StrideLSTMRuntime runtime(std::getenv("STRIDE_LSTM_MODEL_BIN"));
    std::map<uint64_t,stride_lstm::RecurrentState> states;uint64_t pc,line;
    while(std::cin>>pc>>line){auto it=states.find(pc);stride_lstm::RecurrentState prior;if(it!=states.end())prior=it->second;else{prior.hidden.assign(runtime.model().hidden_size(),0);prior.cell=prior.hidden;}
      auto value=runtime.InferState(pc,line,prior);states[pc]=stride_lstm::RecurrentState{value.hidden,value.cell};result(value);std::cout<<std::endl;}
    return 0;
  }
  CACHE cache;Online602 engine("online602",&cache);char op;uint64_t cycle,pc,line;
  while(std::cin>>op>>cycle){current_core_cycle[0]=cycle;if(op=='D'){std::cin>>pc>>line;std::vector<uint64_t> output;engine.invoke_prefetcher(pc,line<<6,0,LOAD,output);}else if(op=='T')engine.service();else if(op=='W')warmup_complete[0]=0;else if(op=='M')warmup_complete[0]=1;else throw std::runtime_error("unknown probe command");snapshot(engine,cache);}
}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}return 0;}
'''


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="e602_", dir="/tmp")
        cls.directory = Path(cls.temporary.name)
        for subdirectory in ("inc", "prefetcher"):
            (cls.directory / subdirectory).mkdir()
        prepare(ROOT, cls.directory)
        for name, source in STUBS.items():
            (cls.directory / "inc" / name).write_text(source)
        probe = cls.directory / "probe.cc"
        probe.write_text(PROBE)
        cls.binary = cls.directory / "probe"
        subprocess.run([
            shutil.which("c++") or "g++", "-std=c++11", "-O1",
            "-I" + str(cls.directory / "inc"), "-I" + str(EXP / "runtime"),
            str(probe), str(EXP / "runtime/online602_engine.cc"),
            str(cls.directory / "prefetcher/stride_lstm_runtime.cc"),
            str(cls.directory / "prefetcher/stride_lstm_model_loader.cc"),
            "-o", str(cls.binary),
        ], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self, hidden=8, seed=7, k=None, name=None):
        trainer = OnlineTrainer(hidden, seed)
        if k is not None:
            with torch.no_grad():
                trainer.model.emit_head.weight.zero_()
                trainer.model.emit_head.bias.copy_(torch.tensor([-1.0, 1.0]))
                trainer.model.log_count_mean.weight.zero_()
                trainer.model.log_count_mean.bias.fill_(math.log(k))
        path = self.directory / (name or "model_h{}_seed{}_k{}.bin".format(hidden, seed, k))
        write_model(path, trainer.model.state_dict(), hidden, 128)
        return trainer.model, path

    def environment(self, model, macs=0, capacity=0, method="frozen", worker_socket=None):
        env = dict(os.environ)
        env.update({"STRIDE_LSTM_MODEL_BIN": str(model), "ONLINE602_METHOD": method,
                    "ONLINE602_MACS": str(macs), "ONLINE602_STATE_CAPACITY": str(capacity),
                    "ONLINE602_OUTPUT_LIMIT": "32"})
        if worker_socket:
            env["ONLINE602_SOCKET"] = str(worker_socket)
        return env

    def run_probe(self, commands, environment, mode=None):
        completed = subprocess.run([str(self.binary)] + ([mode] if mode else []),
                                   input=commands, env=environment, text=True,
                                   capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]

    def test_frozen_python_cpp_parity_h8_h16(self):
        rows = [(11 + index % 5, 1024 + index // 5) for index in range(80)]
        rows += [(11, (1 << 58) - 1), (12, 0), ((1 << 64) - 1, 1 << 57)]
        for hidden in (8, 16):
            for seed in (7, 17):
                model, path = self.model(hidden, seed)
                reference = FrozenStrideLiveModel(model)
                expected = [reference.infer(pc, line) for pc, line in rows]
                actual = self.run_probe("".join("{} {}\n".format(*row) for row in rows), self.environment(path), "forward")
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(expected, actual):
                    for field in ("pc", "line", "emit", "k", "deltas", "addresses"):
                        self.assertEqual(left[field], right[field], (hidden, seed, field))
                    np.testing.assert_allclose(left["hidden"], right["hidden"], atol=2e-6, rtol=0)
                    np.testing.assert_allclose(left["cell"], right["cell"], atol=2e-6, rtol=0)

    def test_finite_state_lru_eviction_and_reallocation(self):
        _, path = self.model()
        commands = "".join("D {} {} {}\n".format(i, i + 1, 1024 + i) for i in range(64))
        commands += "D 64 1 2048\nD 65 65 1088\nD 66 2 4096\n"
        snapshots = self.run_probe(commands, self.environment(path, capacity=64))
        previous = {item["pc"]: item for item in snapshots[63]["states"]}
        final = {item["pc"]: item for item in snapshots[-1]["states"]}
        self.assertEqual(snapshots[-1]["stats"]["state_entries"], 64)
        self.assertEqual(snapshots[-1]["stats"]["state_evictions"], 2)
        self.assertEqual(final[1]["life"], previous[1]["life"])
        self.assertGreater(final[2]["life"], previous[2]["life"])
        self.assertNotIn(3, final)

    def test_delayed_completion_without_new_demand_and_state_visibility(self):
        model, path = self.model(k=2)
        environment = self.environment(path, macs=4, capacity=64)
        initial = self.run_probe("D 0 11 1024\n", environment)[0]
        due = initial["active_due"]
        self.assertGreater(due, 0)
        snapshots = self.run_probe("D 0 11 1024\nT {}\nT {}\nT {}\n".format(due - 1, due, due + 1), environment)
        for snapshot in snapshots[:2]:
            self.assertEqual(snapshot["stats"]["nn_completed"], 0)
            self.assertEqual(snapshot["states"][0]["hidden"], [0.0] * 8)
            self.assertEqual(snapshot["issued"], [])
        reference = FrozenStrideLiveModel(model).infer(11, 1024)
        self.assertEqual(snapshots[2]["stats"]["nn_completed"], 1)
        np.testing.assert_allclose(snapshots[2]["states"][0]["hidden"], reference["hidden"], atol=2e-6, rtol=0)
        self.assertEqual([item[2] for item in snapshots[-1]["issued"]], reference["addresses"])
        self.assertEqual(snapshots[-1]["stats"]["eligible_l2_callbacks"], 1)

    def test_input_decoder_output_limits_visible(self):
        _, path = self.model(k=40)
        snapshots = self.run_probe("".join("D 0 {} 1024\n".format(pc) for pc in range(40)), self.environment(path, macs=4, capacity=64))
        end = snapshots[-1]["stats"]
        self.assertEqual(end["peak_input_queue"], 16)
        self.assertEqual(end["nn_admitted"], 17)
        self.assertEqual(end["inference_dropped"], 23)
        self.assertEqual(end["decoder_dropped_addresses"], 8)
        self.assertEqual(snapshots[-1]["active_result"]["k"], 40)
        self.assertEqual(len(snapshots[-1]["active_result"]["addresses"]), 32)
        due = snapshots[0]["active_due"]
        values = self.run_probe("D 0 11 1024\nT {}\nT {}\nT {}\n".format(due, due, due + 1), self.environment(path, macs=4))
        self.assertEqual(values[1]["stats"]["peak_output_queue"], 32)
        self.assertEqual(len(values[1]["issued"]), 1)
        self.assertEqual(len(values[2]["issued"]), 1, "multiple service calls at same cycle violate output initiation interval")
        self.assertEqual(len(values[3]["issued"]), 2)

    def test_compatibility_warmup_issues_and_trains_nothing(self):
        _, path = self.model(k=2)
        snapshots = self.run_probe("W 0\nD 1 11 1024\nT 1000\nM 1001\nD 1002 11 1025\n", self.environment(path))
        self.assertEqual(snapshots[2]["stats"]["eligible_l2_callbacks"], 0)
        self.assertEqual(snapshots[2]["stats"]["state_entries"], 0)
        self.assertEqual(snapshots[2]["issued"], [])
        self.assertEqual(snapshots[-1]["stats"]["nn_completed"], 1)

    def test_real_worker_publication_preserves_inflight_version_and_state(self):
        _, model_path = self.model()
        socket_path = self.directory / "worker.sock"
        stats_path = self.directory / "worker_stats.json"
        worker = subprocess.Popen([sys.executable, str(EXP / "python/online_worker.py"),
                                   "--socket", str(socket_path), "--stats", str(stats_path)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        probe = None
        try:
            deadline = time.monotonic() + 20
            while not socket_path.exists():
                if worker.poll() is not None or time.monotonic() > deadline:
                    self.fail("worker unavailable: {}".format(worker.communicate(timeout=5)))
                time.sleep(.01)
            probe = subprocess.Popen([str(self.binary)], env=self.environment(model_path, macs=4, capacity=64, method="online", worker_socket=socket_path),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            def command(text):
                probe.stdin.write(text + "\n")
                probe.stdin.flush()
                while True:
                    line = probe.stdout.readline()
                    if not line:
                        self.fail("probe failed: " + probe.stderr.read())
                    if line.startswith("{"):
                        return json.loads(line)
            now = 0
            for index in range(64):
                started = command("D {} 11 {}".format(now, 1024 + index % 40))
                if index == 0:
                    self.assertEqual(started["stats"]["observed_teacher_decisions"], 1)
                    self.assertEqual(started["stats"]["available_supervised_decisions"], 0)
                    before_label = command("T 23")
                    self.assertEqual(before_label["stats"]["available_supervised_decisions"], 0)
                    ready_label = command("T 24")
                    self.assertEqual(ready_label["stats"]["available_supervised_decisions"], 1)
                    self.assertEqual(ready_label["stats"]["nn_completed"], 0)
                now = started["active_due"]
                completed = command("T {}".format(now))
            self.assertEqual(completed["stats"]["published_versions"], 0)
            self.assertEqual(completed["stats"]["update_inflight"], 1)
            publish_due = completed["publish_due"]
            started = command("D {} 11 2048".format(publish_due - 1))
            self.assertEqual(started["stats"]["model_version"], 0)
            prior_state = started["states"][0]
            old_result = started["active_result"]
            published = command("T {}".format(publish_due))
            self.assertEqual(published["stats"]["model_version"], 1)
            self.assertEqual(published["active_result"], old_result)
            self.assertEqual(published["states"][0], prior_state)
            finished = command("T {}".format(started["active_due"]))
            np.testing.assert_array_equal(finished["states"][0]["hidden"], old_result["hidden"])
            np.testing.assert_array_equal(finished["states"][0]["cell"], old_result["cell"])
            self.assertEqual(finished["stats"]["completed_updates"], 1)
            self.assertEqual(finished["stats"]["sample_exposures"], 64)
            self.assertEqual(finished["stats"]["peak_inflight_updates"], 1)
            probe.stdin.close()
            probe.wait(timeout=20)
            self.assertEqual(probe.returncode, 0, probe.stderr.read())
            worker.communicate(timeout=20)
            self.assertEqual(worker.returncode, 0)
            stats = json.loads(stats_path.read_text())
            self.assertGreater(stats["last_gradient_l2"], 0)
            self.assertEqual(stats["versions_completed"], 1)
        finally:
            for process in (probe, worker):
                if process and process.poll() is None:
                    process.terminate()
                    process.communicate(timeout=10)
                if process:
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream and not stream.closed:
                            stream.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
