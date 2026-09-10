#ifndef DYNAMIC602_ENGINE_H
#define DYNAMIC602_ENGINE_H
#include "prefetcher.h"
#include "stride_lstm_runtime.h"
#include "stride.h"
#include <deque>
#include <fstream>
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
  std::string json() const;
 private:
  struct Event {
    uint64_t pc=0,line=0,life=0;
    // Measurement-only timestamps and lifetime history; excluded from deployment SRAM.
    uint64_t arrival_cycle=0,start_cycle=0,prior_updates=0;
    stride_lstm::RecurrentState prior;
  };
  struct Entry { uint64_t life=0,lru=0,observed_updates=0; stride_lstm::RecurrentState state; };
  struct Output { uint64_t pc,line,address; };
  CACHE* cache_;
  std::string method_;
  std::unique_ptr<StridePrefetcher> stride_;
  std::unique_ptr<stride_lstm::StrideLSTMRuntime> forward_;
  std::map<uint64_t,Entry> entries_;
  std::deque<Event> inputs_;
  std::deque<Output> outputs_;
  Event active_event_;
  stride_lstm::InferenceResult result_;
  bool active_=false,failed_=false;
  uint64_t due_=0,life_=0,lru_=0,output_next_=0;
  uint64_t macs_=0,capacity_=0,output_limit_=32;
  mutable std::map<std::string,uint64_t> count_;
  std::ofstream history_log_;
  uint64_t history_log_limit_=1024;
  void log_completion(const Entry& entry);
  uint64_t now() const;
  void begin(Event event);
  uint64_t latency(uint64_t mac,uint64_t nonlinear,uint64_t control) const;
  void peak(const std::string& name,uint64_t value);
};
void dynamic602_service(CACHE* cache);
std::string dynamic602_engine_json(unsigned cpu);
#endif
