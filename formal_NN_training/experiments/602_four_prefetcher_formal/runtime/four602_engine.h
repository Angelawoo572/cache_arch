#ifndef FOUR602_ENGINE_H
#define FOUR602_ENGINE_H
#include "prefetcher.h"
#include "live602_runtime.h"
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
class CACHE;
class Streamer;
class AMPM;
class StridePrefetcher;
class SPP_dev2;
class Four602 : public Prefetcher {
 public:
  Four602(std::string type,CACHE* cache);
  ~Four602();
  void invoke_prefetcher(uint64_t pc,uint64_t address,uint8_t hit,uint8_t access,std::vector<uint64_t>& output);
  void cache_fill(uint64_t address,uint32_t set,uint32_t way,uint8_t prefetch,uint64_t evicted);
  void start_roi();
  void note_native_request(uint64_t pc,uint64_t base,uint64_t address,int fill);
  std::string json() const;
  void print_config();
  void dump_stats();
 private:
  CACHE* cache_;
  std::string policy_,method_,mode_;
  std::unique_ptr<live602::Live602Runtime> model_;
  std::unique_ptr<Streamer> streamer_;
  std::unique_ptr<AMPM> ampm_;
  std::unique_ptr<StridePrefetcher> stride_;
  std::unique_ptr<SPP_dev2> spp_;
  uint64_t width_=5000000,total_=25000000,modal_count_=0,warmup_updates_=0;
  bool roi_=false,capturing_=false;
  std::vector<live602::Action> template_,captured_;
  std::set<uint64_t> native_seen_;
  std::map<std::string,uint64_t> counters_;
  std::map<std::string,std::map<int64_t,uint64_t>> distributions_;
  mutable std::ofstream inputs_;
  uint64_t phase() const;
  void count(const std::string& key,uint64_t n=1);
  void profile(const std::string& name,int64_t value);
  void profile_actions(const std::string& prefix,const std::vector<live602::Action>& actions,uint64_t base);
  void issue(uint64_t pc,uint64_t base,const std::vector<live602::Action>& actions);
  void callback(uint64_t address,bool demand);
  void load_template(const std::string& path);
};
void four602_roi(CACHE*);
void four602_note_request(CACHE*,uint64_t,uint64_t,uint64_t,int);
std::string four602_engine_json(unsigned cpu);
#endif
