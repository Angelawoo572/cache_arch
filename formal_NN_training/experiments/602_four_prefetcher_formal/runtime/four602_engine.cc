#include "four602_engine.h"
#include "dynamic602_hooks.h"
#include "cache.h"
#include "champsim.h"
#include "streamer.h"
#include "ampm.h"
#include "stride.h"
#include "spp_dev2.h"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
uint64_t dynamic602_retired_instructions(unsigned cpu);
// The archived dependency declares this destructor but never defines it: its
// original registry leaked the object. Owning the unchanged native comparator
// requires completing that lifecycle; this does not alter any prediction.
AMPM::~AMPM(){for(auto* entry:page_buffer)delete entry;}
namespace {
std::map<unsigned,Four602*> engines;
std::string env(const char* key,const char* fallback="") {const char* p=std::getenv(key);return p&&*p?p:fallback;}
uint64_t number(const char* key,uint64_t fallback) {std::string s=env(key);if(s.empty())return fallback;if(s.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error(key);return std::stoull(s);}
}
Four602::Four602(std::string type,CACHE* cache):Prefetcher(type),cache_(cache) {
 policy_=env("FOUR602_POLICY","none");method_=env("FOUR602_METHOD","none");mode_=env("FOUR602_MODE","full");
 width_=number("DYNAMIC602_PHASE_INSTRUCTIONS",5000000);total_=number("DYNAMIC602_ROI_INSTRUCTIONS",25000000);
 if(!width_||!total_)throw std::runtime_error("invalid phase/ROI");
 if(method_!="none"&&method_!="monitor"&&method_!="native"&&method_!="student"&&method_!="template")throw std::runtime_error("invalid four602 method");
 if(mode_!="full"&&mode_!="fixed_payload"&&mode_!="reset_state")throw std::runtime_error("invalid four602 mode");
 if(policy_!="none"&&policy_!="stride"&&policy_!="streamer"&&policy_!="ampm"&&policy_!="spp")throw std::runtime_error("invalid four602 policy");
 if(method_=="student") {
  if(policy_=="stride"||policy_=="none")throw std::runtime_error("Stride retains its own per-PC runtime");
  model_.reset(new live602::Live602Runtime(env("FOUR602_MODEL")));
 }
 if(method_=="native") {
  if(policy_=="streamer")streamer_.reset(new Streamer("streamer"));
  else if(policy_=="ampm")ampm_.reset(new AMPM("ampm"));
  else if(policy_=="stride")stride_.reset(new StridePrefetcher("stride"));
  else if(policy_=="spp")spp_.reset(new SPP_dev2("spp_dev2",cache));
  else throw std::runtime_error("native policy missing");
 }
 if(method_=="template"||mode_=="fixed_payload")load_template(env("FOUR602_TEMPLATE"));
 const std::string path=env("FOUR602_INPUT_LOG");
 if(!path.empty()) {inputs_.open(path);if(!inputs_)throw std::runtime_error("input log open failed");inputs_<<"sequence,instructions,cycle,kind,address,completed_updates\n";}
 engines[cache_->cpu]=this;print_config();
}
Four602::~Four602(){engines.erase(cache_->cpu);}
uint64_t Four602::phase()const {return std::min<uint64_t>((total_-1)/width_,dynamic602_retired_instructions(cache_->cpu)/width_);}
void Four602::count(const std::string& key,uint64_t n){counters_[key]+=n;}
void Four602::profile(const std::string& name,int64_t value){++distributions_["phase_"+std::to_string(phase())+"_"+name][value];}
void Four602::load_template(const std::string& path) {
 std::ifstream in(path);std::string token;
 if(!(in>>token>>modal_count_)||token!="modal_count")throw std::runtime_error("bad frozen template count");
 int64_t delta;int fill;
 while(in>>delta>>fill) {if(fill!=2&&fill!=4)throw std::runtime_error("bad frozen template fill");live602::Action a;a.delta=delta;a.fill=fill;template_.push_back(a);}
 if(!in.eof()||template_.empty()||modal_count_>template_.size())throw std::runtime_error("bad frozen template rows");
}
void Four602::start_roi() {
 if(roi_)return;
 uint64_t completed=counters_["completed_updates"],demand=counters_["demand_callbacks"],fill=counters_["fill_callbacks"];
 counters_.clear();distributions_.clear();
 counters_["warmup_completed_updates"]=completed;counters_["warmup_demand_callbacks"]=demand;counters_["warmup_fill_callbacks"]=fill;
 warmup_updates_=completed;roi_=true;
}
void Four602::profile_actions(const std::string& prefix,const std::vector<live602::Action>& actions,uint64_t base) {
 if(!roi_)return;
 profile(prefix+"emit",!actions.empty());profile(prefix+"K",actions.size());
 std::set<uint64_t> seen;
 for(size_t slot=0;slot<actions.size();++slot) {
  const auto& a=actions[slot];const bool current=(a.address>>6)==(base>>6),duplicate=!seen.insert(a.address>>6).second;
  profile(prefix+"delta",a.delta);profile(prefix+"delta_slot_"+std::to_string(slot),a.delta);profile(prefix+"fill",a.fill);
  profile(prefix+"fill_slot_"+std::to_string(slot),a.fill);
  profile(prefix+"delta_fill_"+std::to_string(a.fill)+"_slot_"+std::to_string(slot),a.delta);
  count(prefix+"generated");if(current)count(prefix+"current_line");if(duplicate)count(prefix+"duplicates");
 }
}
void Four602::issue(uint64_t pc,uint64_t base,const std::vector<live602::Action>& actions) {
 profile_actions("payload_",actions,base);
 std::set<uint64_t> seen;
 for(size_t slot=0;slot<actions.size();++slot) {
  const auto& a=actions[slot];const bool duplicate=!seen.insert(a.address>>6).second;
  if(roi_)dynamic602_label_request(cache_,pc,base,a.address,phase(),slot,(base>>6)==(a.address>>6),duplicate,3);
  cache_->prefetch_line(pc,base,a.address,a.fill,0);
 }
}
void Four602::callback(uint64_t address,bool demand) {
 if(inputs_&&roi_)inputs_<<(counters_["demand_callbacks"]+counters_["fill_callbacks"]-1)<<','<<dynamic602_retired_instructions(cache_->cpu)<<','<<current_core_cycle[cache_->cpu]<<','<<(demand?'D':'F')<<','<<(address&~uint64_t(63))<<','<<counters_["completed_updates"]<<'\n';
}
void Four602::invoke_prefetcher(uint64_t pc,uint64_t address,uint8_t hit,uint8_t access,std::vector<uint64_t>& output) {
 if(access!=LOAD)return;
 if(warmup_complete[cache_->cpu]&&!roi_)start_roi();
 count("demand_callbacks");callback(address,true);
 std::vector<live602::Action> actions;
 if(method_=="native") {
  std::vector<uint64_t> native;captured_.clear();native_seen_.clear();capturing_=true;
  if(streamer_)streamer_->invoke_prefetcher(pc,address,hit,access,native);
  if(ampm_)ampm_->invoke_prefetcher(pc,address,hit,access,native);
  if(stride_)stride_->invoke_prefetcher(pc,address,hit,access,native);
  if(spp_)spp_->invoke_prefetcher(pc,address,hit,access,native);
  capturing_=false;
  if(spp_) {profile_actions("raw_",captured_,address);profile_actions("payload_",captured_,address);return;}
  for(uint64_t target:native){live602::Action a;a.address=target;a.delta=int64_t(target>>6)-int64_t(address>>6);a.fill=2;actions.push_back(a);}
 } else if(method_=="student") {
  if(mode_=="reset_state")model_->Reset();
  auto result=model_->Infer(address&~uint64_t(63),true);
  count("completed_updates");count("demand_inferences");count("inference_macs",result.macs);count("inference_nonlinears",result.nonlinears);
  if(roi_){profile("learned_emit",result.emit);profile("learned_K",result.count);}
  actions=result.actions;profile_actions("raw_",actions,address);
  if(mode_=="fixed_payload")for(size_t i=0;i<actions.size();++i){auto t=template_[std::min(i,template_.size()-1)];if(i>=template_.size())count("template_extrapolated_slots");t.address=((address>>6)+uint64_t(t.delta))&((uint64_t(1)<<58)-1);t.address<<=6;actions[i]=t;}
 } else if(method_=="template") {
  for(size_t i=0;i<modal_count_;++i){auto a=template_.at(i);a.address=(((address>>6)+uint64_t(a.delta))&((uint64_t(1)<<58)-1))<<6;actions.push_back(a);}
 }
 if(method_!="student")profile_actions("raw_",actions,address);
 issue(pc,address,actions);
}
void Four602::cache_fill(uint64_t address,uint32_t set,uint32_t way,uint8_t prefetch,uint64_t evicted) {
 if(policy_!="spp")return;
 if(warmup_complete[cache_->cpu]&&!roi_)start_roi();
 // Training saw the raw CACHE L2-fill callback at both actual fill call sites.
 // Native SPP's existing registry calls its FILTER feedback only for PF fills.
 // Preserve that native behavior; do not silently broaden the comparator.
 count("fill_callbacks");callback(evicted,false);
 if(spp_&&prefetch)spp_->cache_fill(address,set,way,prefetch,evicted);
 if(model_) {
  if(mode_=="reset_state")model_->Reset();
  auto result=model_->Infer(evicted&~uint64_t(63),false);
  if(!result.actions.empty()||result.count)throw std::runtime_error("SPP fill emitted actions");
  count("completed_updates");count("fill_inferences");count("inference_macs",result.macs);count("inference_nonlinears",result.nonlinears);
 }
}
void Four602::note_native_request(uint64_t pc,uint64_t base,uint64_t address,int fill) {
 if(!capturing_)return;
 live602::Action a;a.address=address;a.delta=int64_t(address>>6)-int64_t(base>>6);a.fill=fill;
 bool duplicate=!native_seen_.insert(address>>6).second;
 if(roi_)dynamic602_label_request(cache_,pc,base,address,phase(),captured_.size(),(base>>6)==(address>>6),duplicate,3);
 captured_.push_back(a);
}
std::string Four602::json()const {
 // ChampSim retains raw Prefetcher pointers until process exit. Its final
 // statistics call must explicitly drain the measurement stream: relying on
 // this object's destructor loses the buffered final callback records.
 if(inputs_.is_open()){inputs_.flush();if(!inputs_)throw std::runtime_error("four602 callback log flush failed");}
 std::ostringstream out;out<<std::setprecision(17)<<"\"four602_policy\":\""<<policy_<<"\",\"four602_method\":\""<<method_<<"\",\"four602_mode\":\""<<mode_<<"\",\"four602_counters\":{";
 bool first=true;for(auto& p:counters_){if(!first)out<<',';first=false;out<<'"'<<p.first<<"\":"<<p.second;}out<<"},\"four602_distributions\":{";
 first=true;for(auto& p:distributions_){if(!first)out<<',';first=false;out<<'"'<<p.first<<"\":{";bool inner=true;for(auto& q:p.second){if(!inner)out<<',';inner=false;out<<'"'<<q.first<<"\":"<<q.second;}out<<'}';}out<<'}';
 if(model_){auto state=model_->GetState();out<<",\"four602_global_state_entries\":1,\"four602_state_bytes\":"<<4*(state.hidden.size()+state.cell.size())<<",\"four602_state_evictions\":0";}
 return out.str();
}
void Four602::print_config(){std::cout<<"FOUR602 policy="<<policy_<<" method="<<method_<<" mode="<<mode_<<" frozen=1 MAC0=synchronous_global_state no_input_queue no_decoder_limit"<<std::endl;}
void Four602::dump_stats(){std::cout<<json()<<std::endl;}
void four602_roi(CACHE* cache){auto it=engines.find(cache->cpu);if(it!=engines.end())it->second->start_roi();}
void four602_note_request(CACHE* cache,uint64_t pc,uint64_t base,uint64_t address,int fill){if(cache->cache_type!=IS_L2C)return;auto it=engines.find(cache->cpu);if(it!=engines.end())it->second->note_native_request(pc,base,address,fill);}
std::string four602_engine_json(unsigned cpu){auto it=engines.find(cpu);return it==engines.end()?"":it->second->json();}
