#!/usr/bin/env python3
"""Compile/test the actual frozen-weight dynamic inference engine against a small software cache fixture."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[2]
sys.path.insert(0, str(EXP / "runtime"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "formal_NN_training/experiments/602_lstm_stride_live_inference/python"))
from prepare_forward import prepare
from formal_NN_training.common.stride_direct_action_model import CompactPCKeyedHurdleStrideLSTM, FrozenStrideLiveModel
from live_model_format import TENSOR_ORDER, write_model

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
// Conventional comparator fixture only. Production compiles the existing Stride source.
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
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#define private public
#include "dynamic602_engine.h"
#undef private
#include "cache.h"
#include "champsim.h"
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
unsigned char warmup_complete[1]={1};
uint64_t current_core_cycle[1]={0};
uint64_t dynamic602_retired_instructions(unsigned) { return current_core_cycle[0]/2; }
uint64_t dynamic602_total_retired_instructions(unsigned) { return current_core_cycle[0]/2; }
template<class T>void array(const std::vector<T>& values){std::cout<<'[';for(size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
void result(const stride_lstm::InferenceResult& r){std::cout<<"{\"pc\":"<<r.pc<<",\"line\":"<<r.line<<",\"emit\":"<<r.emit<<",\"k\":"<<r.k<<",\"deltas\":";array(r.deltas);std::cout<<",\"addresses\":";array(r.addresses);std::cout<<",\"hidden\":";array(r.hidden);std::cout<<",\"cell\":";array(r.cell);std::cout<<'}';}
std::vector<float> weights(const Dynamic602& engine) {
  std::vector<float> values;
  if(engine.forward_)for(const char* name:{__TENSOR_NAMES__}) {
    const auto& data=engine.forward_->model().tensor(name).data;
    values.insert(values.end(),data.begin(),data.end());
  }
  return values;
}
std::vector<float> initial_weights;
struct RequestLabel {uint64_t pc,base,address,phase;unsigned slot;bool current,duplicate;unsigned teacher;};
std::vector<RequestLabel> request_labels;
void dynamic602_label_request(CACHE*,uint64_t pc,uint64_t base,uint64_t address,uint64_t phase,unsigned slot,bool current,bool duplicate,unsigned teacher){request_labels.push_back(RequestLabel{pc,base,address,phase,slot,current,duplicate,teacher});}
void snapshot(Dynamic602& engine,CACHE& cache){
  std::cout<<"{\"stats\":{"<<engine.json()<<"},\"cycle\":"<<current_core_cycle[0]<<",\"active_due\":"<<engine.due_<<",\"weights_unchanged\":"<<(weights(engine)==initial_weights?"true":"false")<<",\"states\":[";
  bool first=true;for(const auto& item:engine.entries_){if(!first)std::cout<<',';first=false;std::cout<<"{\"pc\":"<<item.first<<",\"life\":"<<item.second.life<<",\"observed_updates\":"<<item.second.observed_updates<<",\"hidden\":";array(item.second.state.hidden);std::cout<<",\"cell\":";array(item.second.state.cell);std::cout<<'}';}
  std::cout<<"],\"issued\":[";first=true;for(const auto& item:cache.requests){if(!first)std::cout<<',';first=false;std::cout<<'['<<item.pc<<','<<item.base<<','<<item.address<<']';}
  std::cout<<"],\"request_labels\":[";first=true;for(const auto& label:request_labels){if(!first)std::cout<<',';first=false;std::cout<<"{\"pc\":"<<label.pc<<",\"base\":"<<label.base<<",\"address\":"<<label.address<<",\"phase\":"<<label.phase<<",\"slot\":"<<label.slot<<",\"current\":"<<(label.current?"true":"false")<<",\"duplicate\":"<<(label.duplicate?"true":"false")<<",\"teacher\":"<<label.teacher<<'}';}
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
  CACHE cache;Dynamic602 engine("dynamic602",&cache);initial_weights=weights(engine);char op;uint64_t cycle,pc,line;
  while(std::cin>>op>>cycle){current_core_cycle[0]=cycle;if(op=='D'){std::cin>>pc>>line;std::vector<uint64_t> output;engine.invoke_prefetcher(pc,line<<6,0,LOAD,output);for(auto address:output)cache.prefetch_line(pc,line<<6,address,FILL_L2,0);}else if(op=='T')engine.service();else if(op=='Z'){engine.result_.addresses.clear();engine.result_.deltas.clear();engine.result_.k=0;engine.result_.emit=1;}else if(op=='W')warmup_complete[0]=0;else if(op=='M')warmup_complete[0]=1;else throw std::runtime_error("unknown probe command");snapshot(engine,cache);}
}catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}return 0;}
'''


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory(prefix="e602_", dir="/tmp")
        cls.directory = Path(cls.temporary.name)
        for subdirectory in ("inc", "prefetcher"):
            (cls.directory / subdirectory).mkdir()
        prepare(ROOT, cls.directory)
        for name, source in STUBS.items():
            (cls.directory / "inc" / name).write_text(source)
        probe = cls.directory / "probe.cc"
        probe.write_text(PROBE.replace("__TENSOR_NAMES__", ",".join(json.dumps(name) for name in TENSOR_ORDER)))
        cls.binary = cls.directory / "probe"
        subprocess.run([
            shutil.which("c++") or "g++", "-std=c++11", "-O1",
            "-I" + str(cls.directory / "inc"), "-I" + str(EXP / "runtime"),
            str(probe), str(EXP / "runtime/dynamic602_engine.cc"),
            str(cls.directory / "prefetcher/stride_lstm_runtime.cc"),
            str(cls.directory / "prefetcher/stride_lstm_model_loader.cc"),
            "-o", str(cls.binary),
        ], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self, hidden=8, seed=7, k=None, name=None):
        torch.manual_seed(seed)
        model = CompactPCKeyedHurdleStrideLSTM(128, hidden).cpu().eval()
        if k is not None:
            with torch.no_grad():
                model.emit_head.weight.zero_()
                model.emit_head.bias.copy_(torch.tensor([-1.0, 1.0]))
                model.log_count_mean.weight.zero_()
                model.log_count_mean.bias.fill_(math.log(k))
        path = self.directory / (name or "model_h{}_seed{}_k{}.bin".format(hidden, seed, k))
        write_model(path, model.state_dict(), hidden, 128)
        return model, path

    def environment(self, model, macs=0, capacity=0, method="frozen"):
        env = dict(os.environ)
        env.update({"STRIDE_LSTM_MODEL_BIN": str(model), "DYNAMIC602_METHOD": method,
                    "DYNAMIC602_MACS": str(macs), "DYNAMIC602_STATE_CAPACITY": str(capacity),
                    "DYNAMIC602_OUTPUT_LIMIT": "32"})
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
        self.assertEqual(final[1]["observed_updates"], 2)
        self.assertEqual(final[2]["observed_updates"], 1)

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

    def test_compatibility_warmup_preserves_empty_predictor(self):
        _, path = self.model(k=2)
        snapshots = self.run_probe("W 0\nD 1 11 1024\nT 1000\nM 1001\nD 1002 11 1025\n", self.environment(path))
        self.assertEqual(snapshots[2]["stats"]["eligible_l2_callbacks"], 0)
        self.assertEqual(snapshots[2]["stats"]["state_entries"], 0)
        self.assertEqual(snapshots[2]["issued"], [])
        self.assertEqual(snapshots[-1]["stats"]["nn_completed"], 1)

    def test_dynamic_predictions_and_states_keep_every_weight_unchanged(self):
        model, path = self.model(k=2)
        for macs in (0, 4, 16):
            environment = self.environment(path, macs=macs, capacity=64)
            commands = "".join("D {} 11 {}\nT {}\nT {}\n".format(index * 10000, 1024 + index, index * 10000 + 9998, index * 10000 + 9999)
                               for index in range(12))
            snapshots = self.run_probe(commands, environment)
            completed = snapshots[2::3]
            self.assertTrue(all(item["weights_unchanged"] for item in snapshots))
            self.assertGreater(len({tuple(item["states"][0]["hidden"]) for item in completed}), 1)
            self.assertGreater(len({tuple(item["states"][0]["cell"]) for item in completed}), 1)
            reference = FrozenStrideLiveModel(model)
            expected = []
            for index in range(12):
                value = reference.infer(11, 1024 + index)
                expected.extend(value["addresses"])
                np.testing.assert_allclose(completed[index]["states"][0]["hidden"], value["hidden"], atol=2e-6, rtol=0)
                np.testing.assert_allclose(completed[index]["states"][0]["cell"], value["cell"], atol=2e-6, rtol=0)
            self.assertEqual([item[2] for item in completed[-1]["issued"]], expected)
            self.assertGreater(len(set(expected)), 1)
            self.assertEqual(completed[-1]["stats"]["nn_completed"], 12)
            self.assertEqual(completed[-1]["stats"]["history_log_rows"], 0)
            self.assertFalse(any(any(word in key for word in ("training", "optimizer", "backward", "publish", "version"))
                                 for key in completed[-1]["stats"]))

    def test_history_counts_completed_lifetimes_and_bounded_log(self):
        _, path = self.model(k=2)
        log = self.directory / "history.jsonl"
        environment = self.environment(path, macs=4, capacity=2)
        environment.update(DYNAMIC602_HISTORY_LOG=str(log), DYNAMIC602_HISTORY_LOG_LIMIT="3")
        initial = self.run_probe("D 0 11 1024\n", environment)[0]
        due = initial["active_due"]
        snapshots = self.run_probe("D 0 11 1024\n" + "D 0 11 1025\n" * 20 +
                                   "T {}\nT {}\nT {}\nT {}\n".format(due, due*2, due*3, due*4), environment)
        queued = snapshots[20]
        self.assertEqual(queued["stats"]["inference_dropped"], 4)
        self.assertEqual(queued["states"][0]["observed_updates"], 0)
        records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual([r["prior_updates"] for r in records], [0, 1, 2])
        self.assertEqual([r["post_updates"] for r in records], [1, 2, 3])
        for before, after in zip(records, records[1:]):
            self.assertEqual(before["new_h"], after["old_h"])
            self.assertEqual(before["new_c"], after["old_c"])
        self.assertGreater(records[1]["start_cycle"] - records[1]["arrival_cycle"], 0)
        self.assertGreater(snapshots[-1]["stats"]["inference_completion_cycles_max"], 0)
        self.assertEqual(snapshots[-1]["stats"]["state_history_completed"], snapshots[-1]["stats"]["nn_completed"])

    def test_shared_full_warmup_conventional_and_teacher_history(self):
        _,path=self.model(k=2)
        commands="W 0\nD 1 11 1024\nD 2 11 1025\nM 3\nD 4 11 1026\n"
        baseline=self.run_probe(commands,self.environment(path,method="stride"))[-1]
        self.assertEqual([r[2] for r in baseline["issued"]],[1026<<6,1027<<6])
        self.assertEqual(baseline["stats"]["shadow_teacher_calls"],1)
        self.assertEqual(baseline["stats"]["teacher_active_events"],1)
        log=self.directory/"shared_warmup_teacher.jsonl"
        env=self.environment(path)
        env["DYNAMIC602_HISTORY_LOG"]=str(log)
        nn=self.run_probe(commands,env)[-1]
        rows=[json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(rows[0]["teacher_active_at_arrival"])
        self.assertEqual(rows[0]["teacher_addresses_at_arrival"],[1026<<6,1027<<6])
        self.assertEqual(nn["states"][0]["observed_updates"],1,"zero NN history must remain empty through shared policy warmup")

    def test_actual_fifo_and_nonlinear_environment(self):
        _, path = self.model(k=2)
        env = self.environment(path, macs=4)
        env.update(DYNAMIC602_INPUT_QUEUE="2", DYNAMIC602_OUTPUT_QUEUE="1", DYNAMIC602_NONLINEAR_CYCLES="9")
        snapshots = self.run_probe("D 0 11 1024\nD 0 11 1025\nD 0 11 1026\nD 0 11 1027\n", env)
        end=snapshots[-1]["stats"]
        self.assertEqual(end["nn_admitted"], 3)
        self.assertEqual(end["inference_dropped"], 1)
        self.assertEqual(end["nonlinear_cycles_per_operation"], 9)
        original=self.run_probe("D 0 11 1024\n",self.environment(path,macs=4))[0]
        self.assertGreater(snapshots[0]["active_due"], original["active_due"])
        done=self.run_probe("D 0 11 1024\nT {}\n".format(snapshots[0]["active_due"]),env)[-1]
        self.assertEqual(done["stats"]["peak_output_queue"],1)
        self.assertEqual(done["stats"]["output_dropped_addresses"],1)

    def test_teacher_snapshot_before_admission_and_silent_output_ablation(self):
        _, path = self.model(k=2)
        env=self.environment(path,macs=4)
        env.update(DYNAMIC602_INPUT_QUEUE="1",DYNAMIC602_SILENT_SUPPRESS="1")
        history=self.directory/"teacher_arrival.jsonl"
        env["DYNAMIC602_HISTORY_LOG"]=str(history)
        initial=self.run_probe("D 0 11 1024\n",env)[0]
        due=initial["active_due"]
        commands="D 0 11 1024\nD 0 11 1025\nD 0 11 1026\nT {}\n".format(due)
        suppressed=self.run_probe(commands,env)[-1]
        rows=[json.loads(line) for line in history.read_text().splitlines()]
        self.assertEqual(suppressed["stats"]["shadow_teacher_calls"],3)
        self.assertEqual(suppressed["stats"]["teacher_active_events"],1)
        self.assertEqual(suppressed["stats"]["inference_dropped"],1)
        self.assertFalse(rows[0]["teacher_active_at_arrival"])
        self.assertEqual(rows[0]["teacher_addresses_at_arrival"],[])
        self.assertEqual(suppressed["stats"]["silent_suppressed_addresses"],2)
        normal_env=dict(env,DYNAMIC602_SILENT_SUPPRESS="0")
        normal=self.run_probe(commands,normal_env)[-1]
        self.assertEqual(normal["states"],suppressed["states"])
        self.assertEqual(normal["stats"]["nn_admitted"],suppressed["stats"]["nn_admitted"])
        self.assertEqual(normal["stats"]["inference_service_cycles"],suppressed["stats"]["inference_service_cycles"])
        self.assertEqual(len(normal["issued"]),1)
        self.assertEqual(suppressed["issued"],[])

    def test_drop_oldest_is_only_queue_policy_change(self):
        _,path=self.model(k=2)
        env=self.environment(path,macs=4)
        env.update(DYNAMIC602_INPUT_QUEUE="1",DYNAMIC602_QUEUE_POLICY="drop_oldest")
        due=self.run_probe("D 0 11 1024\n",env)[0]["active_due"]
        rows=self.run_probe("D 0 11 1024\nD 0 12 1025\nD 0 13 1026\nT {}\n".format(due),env)
        self.assertEqual(rows[-1]["stats"]["queue_evicted_oldest"],1)
        self.assertEqual(rows[-1]["stats"]["shadow_teacher_calls"],3)
        self.assertEqual(rows[-1]["active_result"]["pc"],13)

    def test_state_history_retains_only_completed_warmup_updates(self):
        _,path=self.model(k=2)
        env=self.environment(path,macs=4)
        env.update(DYNAMIC602_WARMUP_INSTRUCTIONS="20000",DYNAMIC602_HISTORY_INSTRUCTIONS="20000")
        rows=self.run_probe("W 0\nD 2 11 1024\nT 10000\nD 10001 11 1025\nM 10002\nT 10003\n",env)
        final=rows[-1]
        self.assertEqual(final["stats"]["warmup_completed_updates"],1)
        self.assertEqual(final["stats"]["warmup_discarded_work"],1)
        self.assertEqual(final["stats"]["nn_completed"],0)
        self.assertEqual(final["stats"]["input_queue"],0)
        self.assertEqual(final["states"][0]["observed_updates"],1)
        self.assertEqual(final["issued"],[])
        self.assertEqual(final["stats"]["shadow_teacher_calls"],0)

    def test_formal_input_log_includes_all_roi_loads_with_reset_occurrences(self):
        _,path=self.model(k=2)
        env=self.environment(path,method="none")
        path=self.directory/"formal_input.csv"
        env.update(DYNAMIC602_INPUT_LOG=str(path),DYNAMIC602_TRACE_NAME="605.mcf_s-994B")
        self.run_probe("W 0\nD 1 11 1024\nM 2\nD 3 11 1024\nD 4 11 1024\n",env)
        import csv
        rows=list(csv.DictReader(path.open()))
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]["trace"],"605.mcf_s-994B")
        self.assertEqual([int(row["pc_line_occ"]) for row in rows],[0,1])
        self.assertEqual([int(row["demand_idx"]) for row in rows],[0,1])

    def test_fixed_payload_keeps_full_forward_state_and_service(self):
        _,path=self.model(k=3)
        environment=self.environment(path,macs=4,capacity=64)
        due=self.run_probe("D 0 11 1024\n",environment)[0]["active_due"]
        command="D 0 11 1024\nT {}\nT {}\nT {}\n".format(due,due+1,due+2)
        learned=self.run_probe(command,environment)[-1]
        fixed=self.run_probe(command,dict(environment,DYNAMIC602_PAYLOAD="fixed_plus2"))[-1]
        self.assertEqual(learned["states"],fixed["states"])
        for key in ("nn_started","nn_completed","inference_macs","inference_nonlinears","inference_service_cycles","state_history_completed","decoder_dropped_addresses"):
            self.assertEqual(learned["stats"][key],fixed["stats"][key],key)
        self.assertEqual([r[2] for r in fixed["issued"]],[1026<<6])
        self.assertTrue(fixed["weights_unchanged"])
        self.assertEqual(fixed["stats"]["mechanism_distributions_exact"]["phase_0_raw_K"],{"3":1})
        self.assertEqual(fixed["stats"]["mechanism_distributions_exact"]["phase_0_payload_K"],{"1":1})
        self.assertEqual(fixed["stats"]["mechanism_distributions_exact"]["phase_0_payload_delta_slot_0"],{"2":1})

    def test_fixed_payload_uses_emit_even_when_learned_list_empty(self):
        _,path=self.model(k=2)
        environment=dict(self.environment(path,macs=4),DYNAMIC602_PAYLOAD="fixed_plus2")
        due=self.run_probe("D 0 11 1024\n",environment)[0]["active_due"]
        rows=self.run_probe("D 0 11 1024\nZ 1\nT {}\n".format(due),environment)
        self.assertEqual(rows[-1]["stats"]["mechanism_distributions_exact"]["phase_0_raw_K"],{"0":1})
        self.assertEqual(rows[-1]["stats"]["mechanism_distributions_exact"]["phase_0_raw_emit"],{"1":1})
        self.assertEqual([r[2] for r in rows[-1]["issued"]],[1026<<6])

    def test_unconditional_and_bounded_frozen_pc_gate(self):
        _,path=self.model(k=2)
        table=self.directory/"pc_gate.tsv"
        table.write_text("pc\tgate\tcount\tpositives\n11\t1\t4\t2\n12\t0\t3\t1\n")
        commands="W 0\nD 1 11 1024\nM 2\nD 3 11 1024\nD 4 12 1025\nD 5 13 1026\n"
        unconditional=self.run_probe(commands,self.environment(path,method="fixed_plus2"))[-1]
        self.assertEqual([r[2] for r in unconditional["issued"]],[1026<<6,1027<<6,1028<<6])
        env=self.environment(path,method="pc_gate_plus2");env["DYNAMIC602_GATE_TABLE"]=str(table)
        gated=self.run_probe(commands,env)[-1]
        self.assertEqual([r[2] for r in gated["issued"]],[1026<<6])
        self.assertEqual(gated["stats"]["pc_gate_seen_events"],2)
        self.assertEqual(gated["stats"]["pc_gate_unseen_events"],1)
        self.assertEqual(gated["stats"]["pc_gate_entries"],2)
        self.assertEqual(gated["stats"]["pc_gate_capacity"],64)
        self.assertEqual(gated["stats"]["pc_gate_table_packed_bits"],4224)
        self.assertEqual(gated["stats"]["nn_completed"],0)
        self.assertEqual(gated["states"],[])
        self.assertEqual(gated["request_labels"][0]["slot"],0)
        self.assertFalse(gated["request_labels"][0]["current"])
        # Exact majority and fixed capacity are validated, rather than silently
        # loading arbitrary gate values or an unbounded PC map.
        for text in ("pc gate count positives\n11 1 3 1\n",
                     "pc gate count positives\n11 1 2 1\n11 0 2 0\n",
                     "pc gate count positives\n"+"".join("{} 1 1 1\n".format(i) for i in range(65))):
            table.write_text(text)
            rejected=subprocess.run([str(self.binary)],input="",env=env,text=True,capture_output=True)
            self.assertNotEqual(rejected.returncode,0)

    def test_exact_phase_slot_current_and_duplicate_diagnostics(self):
        _,path=self.model(k=2)
        # Zero coordinates produce repeated current-line candidates. Diagnostics
        # classify both slots; no list duplicate is removed by the engine.
        model,_=self.model(k=2)
        with torch.no_grad():
            model.action_decoder.delta_head.weight.zero_()
            model.action_decoder.delta_head.bias.zero_()
        path=self.directory/"current_duplicate.bin"
        write_model(path,model.state_dict(),8,128)
        env=self.environment(path);env["DYNAMIC602_PHASE_INSTRUCTIONS"]="5"
        rows=self.run_probe("D 20 11 1024\n",env)
        final=rows[-1]
        self.assertEqual([r[2] for r in final["issued"]],[1024<<6,1024<<6])
        self.assertEqual([r["duplicate"] for r in final["request_labels"]],[False,True])
        self.assertTrue(all(r["current"] for r in final["request_labels"]))
        self.assertEqual(final["stats"]["mechanism_payload_current_line_addresses"],2)
        self.assertEqual(final["stats"]["mechanism_payload_list_duplicate_addresses"],1)
        self.assertEqual(final["stats"]["mechanism_request_profile"],{
            "phase_2/slot_0/current_1/duplicate_0/teacher_2":1,
            "phase_2/slot_1/current_1/duplicate_1/teacher_2":1})
        self.assertEqual(final["stats"]["mechanism_distributions_exact"]["phase_2_raw_K"],{"2":1})

    def test_actual_fitted_pc_tables_when_available(self):
        paths=json.loads(os.environ.get("DYNAMIC602_VERIFY_GATE_TABLES","[]"))
        if not paths:
            self.skipTest("actual three seed tables are verified after full-stream fitting")
        self.assertEqual(len(paths),3)
        results=[]
        for name in paths:
            table=Path(name);before=table.read_bytes()
            lines=before.decode("ascii").splitlines()
            self.assertEqual(lines[0].split(),["pc","gate","count","positives"])
            rows=[tuple(map(int,row.split())) for row in lines[1:]]
            self.assertLessEqual(len(rows),64)
            self.assertEqual(len({row[0] for row in rows}),len(rows))
            unknown=next(pc for pc in range(65) if pc not in {row[0] for row in rows})
            env=self.environment(table,method="pc_gate_plus2")
            env["DYNAMIC602_GATE_TABLE"]=str(table)
            commands="".join("D {} {} 1024\n".format(i,row[0]) for i,row in enumerate(rows))
            commands+="D {} {} 1024\n".format(len(rows),unknown)
            actual=self.run_probe(commands,env)[-1]
            expected=[row[0] for row in rows if row[1]]
            self.assertEqual([row[0] for row in actual["issued"]],expected)
            self.assertEqual([row[2] for row in actual["issued"]],[1026<<6]*len(expected))
            self.assertEqual(actual["stats"]["pc_gate_entries"],len(rows))
            self.assertEqual(actual["stats"]["pc_gate_seen_events"],len(rows))
            self.assertEqual(actual["stats"]["pc_gate_unseen_events"],1)
            self.assertEqual(actual["stats"]["pc_gate_emit_events"],len(expected))
            self.assertEqual(actual["stats"]["nn_completed"],0)
            self.assertEqual(table.read_bytes(),before)
            results.append(dict(table=str(table),entries=len(rows),gate1=len(expected),gate0=len(rows)-len(expected),
                                unknown_pc=unknown,unknown_silent=True,all_loaded_gates_equal=True,
                                file_bytes_unchanged=True,rows=[dict(pc=pc,gate=gate,count=count,positives=positive)
                                                              for pc,gate,count,positive in rows]))
        output=os.environ.get("DYNAMIC602_GATE_TABLE_VERIFY_OUTPUT")
        if output:
            target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(dict(status="verified",scope="real C++ Dynamic602 constructor and every frozen table gate; no performance simulation",results=results),indent=2)+"\n")

    def test_only_frozen_and_conventional_modes_are_available(self):
        _, path = self.model(k=2)
        rejected = subprocess.run([str(self.binary)], input="", env=self.environment(path, method="online"),
                                  text=True, capture_output=True, timeout=10)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("unknown dynamic602 method", rejected.stderr)
        for method in ("none", "stride"):
            snapshots = self.run_probe("D 0 11 1024\nD 1 11 1025\nD 2 11 1026\n", self.environment(path, method=method))
            self.assertEqual(snapshots[-1]["stats"]["nn_completed"], 0)
            self.assertEqual(snapshots[-1]["stats"]["weight_bytes"], 0)
            self.assertEqual(snapshots[-1]["stats"]["state_entries"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
