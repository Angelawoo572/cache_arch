#include "dynamic602_engine.h"
#include "cache.h"
#include "champsim.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
std::map<unsigned,Dynamic602*> engines;
uint64_t envnum(const char* key,uint64_t fallback) {const char* p=std::getenv(key);return p?std::stoull(p):fallback;}
std::string envstr(const char* key,const char* fallback) {const char* p=std::getenv(key);return p?p:fallback;}
uint64_t ceildiv(uint64_t n,uint64_t d){return (n+d-1)/d;}
}

Dynamic602::Dynamic602(std::string type,CACHE* cache):Prefetcher(type),cache_(cache) {
  method_=envstr("DYNAMIC602_METHOD","frozen");
  if(method_!="none"&&method_!="stride"&&method_!="frozen")throw std::runtime_error("unknown dynamic602 method");
  macs_=envnum("DYNAMIC602_MACS",0);capacity_=envnum("DYNAMIC602_STATE_CAPACITY",0);
  output_limit_=envnum("DYNAMIC602_OUTPUT_LIMIT",32);
  if(macs_!=0 && macs_!=4 && macs_!=16)throw std::runtime_error("service case must be 0,4,16");
  if(method_=="stride")stride_.reset(new StridePrefetcher("stride"));
  if(method_=="frozen") {
    forward_.reset(new stride_lstm::StrideLSTMRuntime(envstr("STRIDE_LSTM_MODEL_BIN","")));
    forward_->SetOutputLimit(output_limit_);
  }
  for(const char* key:{"eligible_l2_callbacks","nn_admitted","nn_started","nn_completed","inference_dropped","numerical_failures","decoder_dropped_addresses","output_dropped_addresses","state_evictions","peak_state_entries","peak_input_queue","peak_output_queue","peak_decoded_addresses","inference_macs","inference_nonlinears","inference_service_cycles","inference_scratch_traffic_bytes"})count_[key]=0;
  engines[cache_->cpu]=this;print_config();
}
Dynamic602::~Dynamic602(){engines.erase(cache_->cpu);}
uint64_t Dynamic602::now()const{return current_core_cycle[cache_->cpu];}
void Dynamic602::peak(const std::string& name,uint64_t value){count_[name]=std::max(count_[name],value);}
uint64_t Dynamic602::latency(uint64_t mac,uint64_t nonlinear,uint64_t control)const {
  if(!macs_)return 0;
  // Matched 4*MACs bytes/cycle weights; one nonlinear unit, 4 cycles/op.
  // Dense arithmetic and nonlinear/control phases serialize conservatively.
  return ceildiv(mac,macs_)+4*nonlinear+control;
}
void Dynamic602::invoke_prefetcher(uint64_t pc,uint64_t address,uint8_t hit,uint8_t access,std::vector<uint64_t>& output){
  if(access!=LOAD||!warmup_complete[cache_->cpu])return;
  ++count_["eligible_l2_callbacks"];
  if(method_=="none")return;
  if(method_=="stride"){stride_->invoke_prefetcher(pc,address,hit,access,output);return;}
  service();
  Event event;event.pc=pc;event.line=address>>LOG2_BLOCK_SIZE;
  if(inputs_.size()>=16){++count_["inference_dropped"];return;}
  inputs_.push_back(std::move(event));++count_["nn_admitted"];
  peak("peak_input_queue",inputs_.size());service();
}
void Dynamic602::begin(Event event){
  ++count_["nn_started"];
  auto it=entries_.find(event.pc);
  if(it==entries_.end()){
    if(capacity_ && entries_.size()>=capacity_){
      auto victim=std::min_element(entries_.begin(),entries_.end(),[](const std::pair<const uint64_t,Entry>& a,const std::pair<const uint64_t,Entry>& b){return a.second.lru<b.second.lru;});
      entries_.erase(victim);++count_["state_evictions"];
    }
    Entry entry;entry.life=++life_;entry.state.hidden.assign(forward_->model().hidden_size(),0);entry.state.cell=entry.state.hidden;
    it=entries_.insert(std::make_pair(event.pc,std::move(entry))).first;
  }
  it->second.lru=++lru_;event.life=it->second.life;event.prior=it->second.state;
  peak("peak_state_entries",entries_.size());active_event_=std::move(event);active_=true;failed_=false;
  try{result_=forward_->InferState(active_event_.pc,active_event_.line,active_event_.prior);}
  catch(const std::exception&){failed_=true;++count_["numerical_failures"];result_=stride_lstm::InferenceResult();}
  const uint64_t h=forward_->model().hidden_size(),k=result_.addresses.size();
  peak("peak_decoded_addresses",k);
  const uint64_t mac=128*h+8*h*h+2*h+(result_.emit?h:0)+k*(3*h*h+4*h);
  const uint64_t nonlinear=6*h+(result_.emit?1:0)+k*(3*h+1);
  count_["inference_macs"]+=mac;count_["inference_nonlinears"]+=nonlinear;
  if(!failed_ && result_.k>k)count_["decoder_dropped_addresses"]+=result_.k-k;
  const uint64_t traffic=4*(128+4*h)+k*24;
  count_["inference_scratch_traffic_bytes"]+=traffic;
  const uint64_t cycles=latency(mac,nonlinear,8+ceildiv(entries_.size(),4)+k*4)+(macs_?ceildiv(traffic,4*macs_):0);
  count_["inference_service_cycles"]+=cycles;due_=now()+cycles;
}
void Dynamic602::service(){
  if(!warmup_complete[cache_->cpu]||!forward_)return;
  // A serial engine respects all recurrent dependencies; h/c become visible only here.
  for(;;){
    if(active_&&now()>=due_){
      auto it=entries_.find(active_event_.pc);
      if(it==entries_.end()||it->second.life!=active_event_.life)throw std::runtime_error("state lifetime changed during inference");
      if(!failed_){it->second.state.hidden=result_.hidden;it->second.state.cell=result_.cell;
        for(auto address:result_.addresses){if(outputs_.size()>=32){++count_["output_dropped_addresses"];continue;}outputs_.push_back(Output{active_event_.pc,active_event_.line,address});}}
      ++count_["nn_completed"];peak("peak_output_queue",outputs_.size());
      active_=false;active_event_=Event();result_=stride_lstm::InferenceResult();
    }
    if(!outputs_.empty() && (!macs_ || now()>=output_next_)){
      const Output item=outputs_.front();outputs_.pop_front();
      cache_->prefetch_line(item.pc,item.line<<LOG2_BLOCK_SIZE,item.address,FILL_L2,0);
      output_next_=now()+1;
      if(!macs_)continue;
    }
    if(!active_&&!inputs_.empty()){Event e=std::move(inputs_.front());inputs_.pop_front();begin(std::move(e));if(!macs_)continue;}
    break;
  }
}
std::string Dynamic602::json()const{
  std::ostringstream s;bool first=true;
  auto add=[&](const std::string& key,uint64_t value){if(!first)s<<',';first=false;s<<'"'<<key<<"\":"<<value;};
  for(const auto& item:count_)add(item.first,item.second);
  add("input_queue",inputs_.size());add("output_queue",outputs_.size());add("inference_inflight",active_);add("state_entries",entries_.size());
  const uint64_t h=forward_?forward_->model().hidden_size():0,p=forward_?forward_->model().parameter_count():0;
  add("weight_bytes",p*4);add("h",h);add("macs_per_cycle",macs_);add("weight_bandwidth_bytes_per_cycle",4*macs_);add("state_capacity",capacity_);
  add("state_entry_bytes",h?32+8*h:0);add("state_bytes_peak",count_["peak_state_entries"]*(32+8*h));
  add("state_bytes_configured",capacity_*(32+8*h));add("stride_bytes_configured",stride_?64*32+16:0);
  add("inference_scratch_bytes",h?(128+26*h)*4:0);
  // Scratch reserves all temporary and carried inference FP32 vectors, including
  // active pre-state and staged result h/c; separately account integer payloads.
  add("active_event_metadata_bytes",h?24:0);
  add("decoded_result_bytes_configured",h?16*(output_limit_?output_limit_:32):0);
  add("decoded_result_bytes_peak",16*count_["peak_decoded_addresses"]);
  add("input_queue_bytes_configured",h?16*24:0);add("output_queue_bytes_configured",h?32*24:0);
  add("input_queue_bytes_peak",count_["peak_input_queue"]*24);add("output_queue_bytes_peak",count_["peak_output_queue"]*24);
  return s.str();
}
void Dynamic602::print_config(){std::cout<<"dynamic602 method "<<method_<<" macs "<<macs_<<" state_capacity "<<capacity_<<" output_limit "<<output_limit_<<std::endl;}
void Dynamic602::dump_stats(){std::cout<<"dynamic602_final {"<<json()<<"}"<<std::endl;}
void dynamic602_service(CACHE* cache){auto it=engines.find(cache->cpu);if(it!=engines.end())it->second->service();}
std::string dynamic602_engine_json(unsigned cpu){auto it=engines.find(cpu);return it==engines.end()?"":it->second->json();}
