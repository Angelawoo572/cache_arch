#ifndef DYNAMIC602_ENGINE_H
#define DYNAMIC602_ENGINE_H
#include "prefetcher.h"
#include "stride_lstm_runtime.h"
#include "stride.h"
#include <deque>
#include <fstream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>
class CACHE;
class Dynamic602 : public Prefetcher {
 public:
  Dynamic602(std::string type, CACHE* cache);
  ~Dynamic602();
  void invoke_prefetcher(uint64_t pc,uint64_t address,uint8_t hit,uint8_t access,std::vector<uint64_t>& output);
  void print_config();
  void dump_stats();
  void service();
  void start_roi();
  std::string json() const;
 private:
  struct Event {
    uint64_t pc=0,line=0,life=0;
    // Only pc/line/life are deployment metadata. All other fields measure
    // timing, action overlap or history and never enter the NN.
    uint64_t arrival_cycle=0,start_cycle=0,prior_updates=0,sequence=0;
    bool teacher_active=false;
    std::vector<uint64_t> teacher_addresses;
    stride_lstm::RecurrentState prior;
  };
  struct Entry { uint64_t life=0,lru=0,observed_updates=0; stride_lstm::RecurrentState state; };
  struct Output {
    uint64_t pc,line,address; unsigned category;
    // Diagnostic labels only; modeled deployment payload remains pc/line/address.
    uint64_t phase; unsigned slot; bool current_line,list_duplicate;
  };
  struct ShadowEntry { uint64_t pc,line; int32_t stride; };
  CACHE* cache_;
  std::string method_,stride_mode_,page_policy_,queue_policy_,payload_mode_;
  std::map<uint64_t,bool> gate_table_;
  uint64_t phase_instructions_=5000000,roi_instructions_=25000000;
  std::unique_ptr<StridePrefetcher> stride_,shadow_;
  // Source-equivalent tracker for forward controls, including page-edge events.
  // It is not a neural feature and does not participate in neural decisions.
  std::list<ShadowEntry> shadow_trackers_;
  std::unique_ptr<stride_lstm::StrideLSTMRuntime> forward_;
  std::map<uint64_t,Entry> entries_;
  std::deque<Event> inputs_;
  std::deque<Output> outputs_;
  Event active_event_;
  stride_lstm::InferenceResult result_;
  bool active_=false,failed_=false,roi_started_=false,silent_suppress_=false;
  uint64_t due_=0,life_=0,lru_=0,output_next_=0,sequence_=0;
  uint64_t macs_=0,capacity_=0,output_limit_=32,input_capacity_=16,output_capacity_=32;
  uint64_t nonlinear_cycles_=4,bandwidth_=0,warmup_instructions_=0,history_instructions_=0;
  uint64_t keep_ppm_=1000000,max_requests_per_event_=0;
  double min_emit_margin_=0;
  mutable std::map<std::string,uint64_t> count_;
  std::map<std::string,std::map<uint64_t,uint64_t>> hist_;
  std::map<std::string,std::map<int64_t,uint64_t>> mechanism_distributions_;
  std::map<std::string,uint64_t> mechanism_requests_;
  std::map<uint64_t,std::pair<uint64_t,uint64_t>> pc_history_,warmup_pc_history_;
  std::ofstream history_log_,input_log_;
  std::string trace_name_;
  std::map<std::pair<uint64_t,uint64_t>,uint64_t> input_occurrences_;
  uint64_t history_log_limit_=1024;
  void log_completion(const Entry& entry);
  uint64_t now() const;
  bool observing_warmup() const;
  void begin(Event event);
  uint64_t latency(uint64_t mac,uint64_t nonlinear,uint64_t control) const;
  void peak(const std::string& name,uint64_t value);
  void sample(const std::string& name,uint64_t value);
  int32_t track_stride(uint64_t pc,uint64_t line);
  bool page_allowed(uint64_t base,uint64_t address) const;
  void append_result();
  void load_gate_table(const std::string& path);
  uint64_t phase() const;
  void profile_value(uint64_t phase,const std::string& name,int64_t value);
  void profile_payload(uint64_t line,const std::vector<uint64_t>& addresses,
                       const std::vector<int64_t>& deltas,const std::vector<uint64_t>& teacher,
                       uint64_t phase,bool raw);
  static unsigned request_category(uint64_t address,const std::vector<uint64_t>& teacher);
};
void dynamic602_service(CACHE* cache);
void dynamic602_roi(CACHE* cache);
std::string dynamic602_engine_json(unsigned cpu);
#endif
