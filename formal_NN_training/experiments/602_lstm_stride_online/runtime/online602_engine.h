#ifndef ONLINE602_ENGINE_H
#define ONLINE602_ENGINE_H
#include "prefetcher.h"
#include "stride_lstm_runtime.h"
#include "stride.h"
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>
class CACHE;
class Online602 : public Prefetcher {
 public:
  Online602(std::string type, CACHE* cache);
  ~Online602();
  void invoke_prefetcher(uint64_t pc,uint64_t address,uint8_t hit,uint8_t access,std::vector<uint64_t>& output);
  void print_config();
  void dump_stats();
  void service();
  std::string json() const;
 private:
  struct Event {
    uint64_t pc=0,line=0,life=0,label_ready=0;
    bool supervised=false;
    std::vector<uint64_t> actions;
    stride_lstm::RecurrentState prior;
  };
  struct Entry { uint64_t life=0,lru=0; stride_lstm::RecurrentState state; };
  struct Output { uint64_t pc,line,address; };
  struct Label { uint64_t ready,positive,actions; };
  CACHE* cache_;
  std::string method_;
  std::unique_ptr<StridePrefetcher> teacher_;
  std::unique_ptr<stride_lstm::StrideLSTMRuntime> forward_;
  std::map<uint64_t,Entry> entries_;
  std::deque<Event> inputs_,training_;
  std::deque<Output> outputs_;
  std::deque<Label> labels_;
  Event active_event_;
  stride_lstm::InferenceResult result_;
  bool active_=false,failed_=false,update_=false,received_=false;
  uint64_t due_=0,train_due_=0,publish_due_=0,life_=0,lru_=0,version_=0,output_next_=0;
  uint64_t macs_=0,capacity_=0,output_limit_=32,training_n_=0;
  int socket_=-1;
  std::vector<float> publication_;
  mutable std::map<std::string,uint64_t> count_;
  uint64_t now() const;
  void begin(Event event);
  void train();
  void receive();
  uint64_t latency(uint64_t mac,uint64_t nonlinear,uint64_t control) const;
  void peak(const std::string& name,uint64_t value);
};
void online602_service(CACHE* cache);
std::string online602_engine_json(unsigned cpu);
#endif
