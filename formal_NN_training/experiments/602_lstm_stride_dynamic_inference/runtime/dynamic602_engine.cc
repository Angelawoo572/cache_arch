#include "dynamic602_engine.h"
#include "cache.h"
#include "champsim.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <set>
#include <stdexcept>

uint64_t dynamic602_retired_instructions(unsigned cpu);
uint64_t dynamic602_total_retired_instructions(unsigned cpu);
// Measurement context never enters predictor features or scheduling. The vector
// path is consumed by the following cache request; the NN path immediately.
void dynamic602_label_request(CACHE*,uint64_t ip,uint64_t base,uint64_t address,
                             uint64_t phase,unsigned slot,bool current_line,
                             bool list_duplicate,unsigned teacher_category);
namespace {
std::map<unsigned,Dynamic602*> engines;
uint64_t envnum(const char* key,uint64_t fallback) {
  const char* p=std::getenv(key);if(!p||!*p)return fallback;
  std::string value(p);size_t end=0;
  if(value[0]=='-')throw std::runtime_error(std::string("negative ")+key);
  const uint64_t number=std::stoull(value,&end);
  if(end!=value.size())throw std::runtime_error(std::string("invalid ")+key);
  return number;
}
std::string envstr(const char* key,const char* fallback) {const char* p=std::getenv(key);return p&&*p?p:fallback;}
uint64_t ceildiv(uint64_t n,uint64_t d){return n/d+(n%d!=0);}
uint64_t mix(uint64_t n){n+=0x9e3779b97f4a7c15ULL;n=(n^(n>>30))*0xbf58476d1ce4e5b9ULL;n=(n^(n>>27))*0x94d049bb133111ebULL;return n^(n>>31);}
const char* categories[]={"teacher_overlap","teacher_active_extra","teacher_silent_extra"};
}

Dynamic602::Dynamic602(std::string type,CACHE* cache):Prefetcher(type),cache_(cache) {
  method_=envstr("DYNAMIC602_METHOD","frozen");
  if(method_!="none"&&method_!="stride"&&method_!="frozen"&&method_!="fixed_plus2"&&method_!="pc_gate_plus2")throw std::runtime_error("unknown dynamic602 method");
  payload_mode_=envstr("DYNAMIC602_PAYLOAD","learned");
  if(payload_mode_!="learned"&&payload_mode_!="fixed_plus2")throw std::runtime_error("invalid dynamic602 payload");
  phase_instructions_=envnum("DYNAMIC602_PHASE_INSTRUCTIONS",5000000);
  roi_instructions_=envnum("DYNAMIC602_ROI_INSTRUCTIONS",25000000);
  if(!phase_instructions_||!roi_instructions_)throw std::runtime_error("zero mechanism phase/ROI width");
  if(method_=="pc_gate_plus2")load_gate_table(envstr("DYNAMIC602_GATE_TABLE",""));
  stride_mode_=envstr("DYNAMIC602_STRIDE_MODE","legacy");
  if(stride_mode_!="legacy"&&stride_mode_!="remove_current"&&stride_mode_!="forward1"&&stride_mode_!="forward2")throw std::runtime_error("unknown stride mode");
  page_policy_=envstr("DYNAMIC602_PAGE_POLICY","cross_page");
  if(page_policy_!="same_page"&&page_policy_!="cross_page")throw std::runtime_error("invalid page policy");
  queue_policy_=envstr("DYNAMIC602_QUEUE_POLICY","fifo");
  if(queue_policy_!="fifo"&&queue_policy_!="drop_oldest")throw std::runtime_error("invalid input queue policy");
  macs_=envnum("DYNAMIC602_MACS",0);capacity_=envnum("DYNAMIC602_STATE_CAPACITY",0);
  output_limit_=envnum("DYNAMIC602_OUTPUT_LIMIT",32);
  input_capacity_=envnum("DYNAMIC602_INPUT_QUEUE",16);output_capacity_=envnum("DYNAMIC602_OUTPUT_QUEUE",32);
  nonlinear_cycles_=envnum("DYNAMIC602_NONLINEAR_CYCLES",4);
  bandwidth_=envnum("DYNAMIC602_BANDWIDTH_BYTES_PER_CYCLE",4*macs_);
  warmup_instructions_=envnum("DYNAMIC602_WARMUP_INSTRUCTIONS",0);
  history_instructions_=envnum("DYNAMIC602_HISTORY_INSTRUCTIONS",0);
  keep_ppm_=envnum("DYNAMIC602_KEEP_PPM",1000000);
  max_requests_per_event_=envnum("DYNAMIC602_MAX_REQUESTS_PER_EVENT",0);
  silent_suppress_=envnum("DYNAMIC602_SILENT_SUPPRESS",0)!=0;
  min_emit_margin_=std::stod(envstr("DYNAMIC602_MIN_EMIT_MARGIN","0"));
  if(!input_capacity_||!output_capacity_||keep_ppm_>1000000||!std::isfinite(min_emit_margin_)||min_emit_margin_<0||history_instructions_>warmup_instructions_||(macs_&&!bandwidth_))throw std::runtime_error("invalid dynamic602 resource or output configuration");
  if(method_=="stride")stride_.reset(new StridePrefetcher("stride"));
  // This independent observer receives ALL eligible arrivals, even those later
  // dropped by NN admission. It is never queried by the neural forward path.
  shadow_.reset(new StridePrefetcher("stride"));
  if(method_=="frozen") {
    forward_.reset(new stride_lstm::StrideLSTMRuntime(envstr("STRIDE_LSTM_MODEL_BIN","")));
    forward_->SetOutputLimit(output_limit_);
  }
  for(const char* key:{"eligible_l2_callbacks","nn_admitted","nn_started","nn_completed","inference_dropped","numerical_failures","decoder_dropped_addresses","output_dropped_addresses","state_evictions","peak_state_entries","peak_input_queue","peak_output_queue","peak_decoded_addresses","inference_macs","inference_nonlinears","inference_service_cycles","inference_scratch_traffic_bytes","inference_weight_read_bytes","state_history_prior_sum","state_history_prior_max","state_history_zero_prior","state_history_completed","inference_wait_cycles_sum","inference_wait_cycles_max","inference_completion_cycles_sum","inference_completion_cycles_max","inference_completed_service_cycles_sum","inference_completed_service_cycles_max","history_log_rows","shadow_teacher_calls","teacher_active_events","teacher_silent_events","generated_addresses","cache_request_attempts","cache_request_accepted","cache_request_rejected","page_filtered_addresses","silent_suppressed_addresses","budget_suppressed_addresses","queue_evicted_oldest"})count_[key]=0;
  for(const char* key:{"mechanism_raw_generated_addresses","mechanism_raw_current_line_addresses","mechanism_raw_list_duplicate_addresses","mechanism_payload_generated_addresses","mechanism_payload_current_line_addresses","mechanism_payload_list_duplicate_addresses","pc_gate_seen_events","pc_gate_unseen_events","pc_gate_emit_events"})count_[key]=0;
  for(const char* category:categories)for(const char* suffix:{"_generated","_attempted","_accepted","_rejected"})count_[std::string(category)+suffix]=0;
  const std::string history_path=envstr("DYNAMIC602_HISTORY_LOG","");
  history_log_limit_=envnum("DYNAMIC602_HISTORY_LOG_LIMIT",1024);
  if(!history_path.empty()) {history_log_.open(history_path);if(!history_log_)throw std::runtime_error("cannot open history log");history_log_<<std::setprecision(9);}
  const std::string input_path=envstr("DYNAMIC602_INPUT_LOG","");
  trace_name_=envstr("DYNAMIC602_TRACE_NAME","602.gcc_s-734B");
  if(trace_name_.find_first_of(",\r\n")!=std::string::npos)throw std::runtime_error("invalid input trace name");
  if(!input_path.empty()){input_log_.open(input_path);if(!input_log_)throw std::runtime_error("cannot open input log");input_log_<<"trace,demand_idx,cycle,pc,line,pc_line_occ\n";}
  engines[cache_->cpu]=this;print_config();
}
Dynamic602::~Dynamic602(){engines.erase(cache_->cpu);}
uint64_t Dynamic602::now()const{return current_core_cycle[cache_->cpu];}
void Dynamic602::peak(const std::string& name,uint64_t value){count_[name]=std::max(count_[name],value);}
void Dynamic602::sample(const std::string& name,uint64_t value){
  // Upper bounds 0,1,3,7,...; every event is represented, including long tails.
  uint64_t upper=0;while(upper<value){if(upper>(std::numeric_limits<uint64_t>::max()-1)/2){upper=std::numeric_limits<uint64_t>::max();break;}upper=upper*2+1;}
  ++hist_[name][upper];
}
bool Dynamic602::observing_warmup()const {
  return history_instructions_ && dynamic602_total_retired_instructions(cache_->cpu)>=warmup_instructions_-history_instructions_;
}
bool Dynamic602::page_allowed(uint64_t base,uint64_t address)const{return page_policy_=="cross_page"||(base>>12)==(address>>12);}
uint64_t Dynamic602::latency(uint64_t mac,uint64_t nonlinear,uint64_t control)const {
  if(!macs_)return 0;
  // One serial engine: dense work and resident-weight reads overlap; nonlinear,
  // control and scratch transfer phases serialize. II equals service duration.
  return std::max(ceildiv(mac,macs_),ceildiv(4*mac,bandwidth_))+nonlinear_cycles_*nonlinear+control;
}
void Dynamic602::start_roi(){
  if(roi_started_)return;
  const uint64_t warmup_completed=count_["nn_completed"],warmup_evictions=count_["state_evictions"];
  const uint64_t discarded=inputs_.size()+active_,discarded_outputs=outputs_.size();
  inputs_.clear();outputs_.clear();active_=false;failed_=false;active_event_=Event();result_=stride_lstm::InferenceResult();due_=now();output_next_=now();
  // Retain completed h/c only. A warmup inference still in service is discarded.
  if(!history_instructions_){entries_.clear();life_=0;lru_=0;}
  for(auto& counter:count_)counter.second=0;
  count_["warmup_completed_updates"]=warmup_completed;count_["warmup_state_evictions"]=warmup_evictions;
  count_["warmup_discarded_work"]=discarded;count_["warmup_discarded_outputs"]=discarded_outputs;
  count_["peak_state_entries"]=entries_.size();
  warmup_pc_history_=pc_history_;
  hist_.clear();mechanism_distributions_.clear();mechanism_requests_.clear();pc_history_.clear();input_occurrences_.clear();sequence_=0;
  // Conventional and shadow Stride retain the same full warmup history in
  // every arm. Only NN h/c visibility and NN pending work vary by treatment.
  roi_started_=true;
}
void Dynamic602::load_gate_table(const std::string& path){
  std::ifstream file(path);if(!file)throw std::runtime_error("cannot open frozen PC gate table");
  std::string row;std::getline(file,row);std::istringstream header(row);
  std::string a,b,c,d,extra;
  if(!(header>>a>>b>>c>>d)||a!="pc"||b!="gate"||c!="count"||d!="positives"||(header>>extra))throw std::runtime_error("invalid frozen PC gate header");
  auto integer=[](const std::string& text){
    if(text.empty()||text.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("nondecimal frozen PC gate field");
    size_t end=0;uint64_t value=std::stoull(text,&end);if(end!=text.size())throw std::runtime_error("invalid frozen PC gate integer");return value;
  };
  while(std::getline(file,row)){
    std::istringstream fields(row);if(!(fields>>a>>b>>c>>d)||(fields>>extra))throw std::runtime_error("invalid frozen PC gate row");
    const uint64_t pc=integer(a),gate=integer(b),count=integer(c),positive=integer(d);
    if(gate>1||!count||positive>count||gate!=(positive>=count/2+count%2))throw std::runtime_error("frozen PC gate majority mismatch");
    if(!gate_table_.emplace(pc,gate!=0).second||gate_table_.size()>64)throw std::runtime_error("frozen PC gate duplicate or capacity exceeded");
  }
}
uint64_t Dynamic602::phase()const{return std::min<uint64_t>((roi_instructions_-1)/phase_instructions_,dynamic602_retired_instructions(cache_->cpu)/phase_instructions_);}
void Dynamic602::profile_value(uint64_t phase,const std::string& name,int64_t value){
  ++mechanism_distributions_["phase_"+std::to_string(phase)+"_"+name][value];
}
unsigned Dynamic602::request_category(uint64_t address,const std::vector<uint64_t>& teacher){
  return std::find(teacher.begin(),teacher.end(),address)!=teacher.end()?0:teacher.empty()?2:1;
}
void Dynamic602::profile_payload(uint64_t line,const std::vector<uint64_t>& addresses,
                                const std::vector<int64_t>& deltas,const std::vector<uint64_t>& teacher,
                                uint64_t phase,bool raw){
  if(!roi_started_)return;
  const std::string prefix=raw?"raw_":"payload_";
  if(!raw){profile_value(phase,prefix+"emit",!addresses.empty());profile_value(phase,prefix+"K",addresses.size());}
  std::set<uint64_t> seen;
  for(size_t slot=0;slot<addresses.size();++slot){
    const uint64_t address=addresses[slot];const bool duplicate=!seen.insert(address).second,current=address==(line<<LOG2_BLOCK_SIZE);
    const int64_t delta=slot<deltas.size()?deltas[slot]:static_cast<int64_t>(address>>LOG2_BLOCK_SIZE)-static_cast<int64_t>(line);
    profile_value(phase,prefix+"delta_slot_"+std::to_string(slot),delta);
    if(raw){++count_["mechanism_raw_generated_addresses"];if(current)++count_["mechanism_raw_current_line_addresses"];if(duplicate)++count_["mechanism_raw_list_duplicate_addresses"];}
    else {
      ++count_["mechanism_payload_generated_addresses"];if(current)++count_["mechanism_payload_current_line_addresses"];if(duplicate)++count_["mechanism_payload_list_duplicate_addresses"];
      const std::string key="phase_"+std::to_string(phase)+"/slot_"+std::to_string(slot)+"/current_"+std::to_string(current)+"/duplicate_"+std::to_string(duplicate)+"/teacher_"+std::to_string(request_category(address,teacher));
      ++mechanism_requests_[key];
    }
  }
}
int32_t Dynamic602::track_stride(uint64_t pc,uint64_t line){
  auto it=std::find_if(shadow_trackers_.begin(),shadow_trackers_.end(),[pc](const ShadowEntry& e){return e.pc==pc;});
  if(it==shadow_trackers_.end()){if(shadow_trackers_.size()>=64)shadow_trackers_.pop_back();shadow_trackers_.push_front(ShadowEntry{pc,line,0});return 0;}
  // Match the inherited Stride's int32 stride conversion and zero-stride return.
  int32_t stride=line>it->line?static_cast<int32_t>(line-it->line):static_cast<int32_t>(0U-static_cast<uint32_t>(it->line-line));
  if(!stride)return 0;
  ShadowEntry entry{pc,line,stride};shadow_trackers_.erase(it);shadow_trackers_.push_front(entry);return stride;
}
void Dynamic602::invoke_prefetcher(uint64_t pc,uint64_t address,uint8_t hit,uint8_t access,std::vector<uint64_t>& output){
  if(access!=LOAD)return;
  const bool measured=warmup_complete[cache_->cpu]!=0;
  if(measured&&!roi_started_)start_roi();
  service();
  if(measured){++count_["eligible_l2_callbacks"];if(input_log_){const uint64_t occurrence=input_occurrences_[std::make_pair(pc,address>>LOG2_BLOCK_SIZE)]++;input_log_<<trace_name_<<','<<(count_["eligible_l2_callbacks"]-1)<<','<<now()<<','<<pc<<','<<(address>>LOG2_BLOCK_SIZE)<<','<<occurrence<<'\n';}}
  std::vector<uint64_t> teacher;
  shadow_->invoke_prefetcher(pc,address,hit,access,teacher);
  if(measured){++count_["shadow_teacher_calls"];++count_[teacher.empty()?"teacher_silent_events":"teacher_active_events"];}
  const int32_t current_stride=track_stride(pc,address>>LOG2_BLOCK_SIZE);
  if(method_=="none")return;
  if(method_=="fixed_plus2"||method_=="pc_gate_plus2"){
    if(!measured)return;
    bool emit=true;
    if(method_=="pc_gate_plus2"){
      auto gate=gate_table_.find(pc);emit=gate!=gate_table_.end()&&gate->second;
      ++count_[gate==gate_table_.end()?"pc_gate_unseen_events":"pc_gate_seen_events"];
      if(emit)++count_["pc_gate_emit_events"];
    }
    std::vector<uint64_t> candidates;if(emit)candidates.push_back((((address>>LOG2_BLOCK_SIZE)+2)&((1ULL<<58)-1))<<LOG2_BLOCK_SIZE);
    const uint64_t p=phase();profile_payload(address>>LOG2_BLOCK_SIZE,candidates,std::vector<int64_t>(candidates.size(),2),teacher,p,false);
    for(auto candidate:candidates){++count_["generated_addresses"];if(!page_allowed(address,candidate)){++count_["page_filtered_addresses"];continue;}
      dynamic602_label_request(cache_,pc,address,candidate,p,0,(candidate>>LOG2_BLOCK_SIZE)==(address>>LOG2_BLOCK_SIZE),false,request_category(candidate,teacher));output.push_back(candidate);}
    return;
  }
  if(method_=="stride"){
    std::vector<uint64_t> candidates;stride_->invoke_prefetcher(pc,address,hit,access,candidates);
    if(!measured)return; // identical noPF cache warmup, conventional state retained
    if(stride_mode_=="remove_current")candidates.erase(std::remove(candidates.begin(),candidates.end(),address),candidates.end());
    else if(stride_mode_=="forward1"||stride_mode_=="forward2"){
      const bool active=!candidates.empty();candidates.clear();
      if(active)for(unsigned degree=1;degree<=(stride_mode_=="forward2"?2U:1U);++degree){
        const uint64_t line=((address>>LOG2_BLOCK_SIZE)+static_cast<uint64_t>(static_cast<int64_t>(current_stride)*degree))&((1ULL<<58)-1);
        if(page_allowed(address,line<<LOG2_BLOCK_SIZE))candidates.push_back(line<<LOG2_BLOCK_SIZE);
      }
    }
    const uint64_t p=phase();profile_payload(address>>LOG2_BLOCK_SIZE,candidates,std::vector<int64_t>(),teacher,p,false);
    std::set<uint64_t> seen;
    for(size_t slot=0;slot<candidates.size();++slot){const uint64_t candidate=candidates[slot];const bool duplicate=!seen.insert(candidate).second;
      if(page_allowed(address,candidate)){dynamic602_label_request(cache_,pc,address,candidate,p,slot,(candidate>>LOG2_BLOCK_SIZE)==(address>>LOG2_BLOCK_SIZE),duplicate,request_category(candidate,teacher));output.push_back(candidate);++count_["generated_addresses"];}else ++count_["page_filtered_addresses"];}
    return;
  }
  if(!measured&&!observing_warmup())return;
  Event event;event.pc=pc;event.line=address>>LOG2_BLOCK_SIZE;event.arrival_cycle=now();event.sequence=++sequence_;event.teacher_active=!teacher.empty();event.teacher_addresses=std::move(teacher);
  if(inputs_.size()>=input_capacity_){
    ++count_["inference_dropped"];
    if(queue_policy_=="fifo")return;
    sample("dropped_queue_age",now()-inputs_.front().arrival_cycle);inputs_.pop_front();++count_["queue_evicted_oldest"];
  }
  inputs_.push_back(std::move(event));++count_["nn_admitted"];peak("peak_input_queue",inputs_.size());service();
}
void Dynamic602::begin(Event event){
  ++count_["nn_started"];
  auto it=entries_.find(event.pc);
  if(it==entries_.end()){
    if(capacity_&&entries_.size()>=capacity_){auto victim=std::min_element(entries_.begin(),entries_.end(),[](const std::pair<const uint64_t,Entry>& a,const std::pair<const uint64_t,Entry>& b){return a.second.lru<b.second.lru;});sample("evicted_state_completed_updates",victim->second.observed_updates);++pc_history_[victim->first].second;entries_.erase(victim);++count_["state_evictions"];}
    Entry entry;entry.life=++life_;entry.state.hidden.assign(forward_->model().hidden_size(),0);entry.state.cell=entry.state.hidden;it=entries_.insert(std::make_pair(event.pc,std::move(entry))).first;
  }
  it->second.lru=++lru_;event.life=it->second.life;event.prior=it->second.state;event.start_cycle=now();event.prior_updates=it->second.observed_updates;
  count_["inference_wait_cycles_sum"]+=now()-event.arrival_cycle;peak("inference_wait_cycles_max",now()-event.arrival_cycle);sample("queue_wait_cycles",now()-event.arrival_cycle);
  peak("peak_state_entries",entries_.size());active_event_=std::move(event);active_=true;failed_=false;
  try{result_=forward_->InferState(active_event_.pc,active_event_.line,active_event_.prior);}catch(const std::exception&){failed_=true;++count_["numerical_failures"];result_=stride_lstm::InferenceResult();}
  const uint64_t h=forward_->model().hidden_size(),k=result_.addresses.size();peak("peak_decoded_addresses",k);
  const uint64_t mac=128*h+8*h*h+2*h+(result_.emit?h:0)+k*(3*h*h+4*h),nonlinear=6*h+(result_.emit?1:0)+k*(3*h+1);
  count_["inference_macs"]+=mac;count_["inference_nonlinears"]+=nonlinear;count_["inference_weight_read_bytes"]+=4*mac;
  if(!failed_&&result_.k>k)count_["decoder_dropped_addresses"]+=result_.k-k;
  const uint64_t traffic=4*(128+4*h)+k*24;count_["inference_scratch_traffic_bytes"]+=traffic;
  const uint64_t duration=latency(mac,nonlinear,8+ceildiv(entries_.size(),4)+k*4)+(macs_?ceildiv(traffic,bandwidth_):0);
  count_["inference_service_cycles"]+=duration;sample("scheduled_service_cycles",duration);due_=now()+duration;
}
void Dynamic602::append_result(){
  const uint64_t p=roi_started_?phase():0;
  if(roi_started_){profile_value(p,"raw_emit",result_.emit);profile_value(p,"raw_K",result_.k);}
  profile_payload(active_event_.line,result_.addresses,result_.deltas,active_event_.teacher_addresses,p,true);
  std::vector<uint64_t> fixed_addresses;std::vector<int64_t> fixed_deltas;
  // Full learned forward, decoder work, completion and h/c have already run.
  // The gate is result.emit, including the empty learned-payload boundary case.
  if(payload_mode_=="fixed_plus2"&&result_.emit){fixed_addresses.push_back(((active_event_.line+2)&((1ULL<<58)-1))<<LOG2_BLOCK_SIZE);fixed_deltas.push_back(2);}
  const std::vector<uint64_t>& addresses=payload_mode_=="learned"?result_.addresses:fixed_addresses;
  const std::vector<int64_t>& deltas=payload_mode_=="learned"?result_.deltas:fixed_deltas;
  profile_payload(active_event_.line,addresses,deltas,active_event_.teacher_addresses,p,false);
  uint64_t retained=0;std::set<uint64_t> seen;
  for(size_t slot=0;slot<addresses.size();++slot){
    const uint64_t address=addresses[slot];const bool duplicate=!seen.insert(address).second,current=address==(active_event_.line<<LOG2_BLOCK_SIZE);
    const unsigned category=request_category(address,active_event_.teacher_addresses);
    ++count_["generated_addresses"];++count_[std::string(categories[category])+"_generated"];
    // Warmup changes only h/c: never insert a prefetch into the warmed cache.
    if(!roi_started_)continue;
    if(!page_allowed(active_event_.line<<LOG2_BLOCK_SIZE,address)){++count_["page_filtered_addresses"];continue;}
    // This output-only ablation preserves the original decode work, admission,
    // scheduled completion, queue pressure up to completion, and h/c update.
    if(silent_suppress_&&!active_event_.teacher_active){++count_["silent_suppressed_addresses"];continue;}
    if(result_.emit_margin<=min_emit_margin_||(max_requests_per_event_&&retained>=max_requests_per_event_)||mix(active_event_.pc^mix(active_event_.line)^mix(active_event_.sequence)^mix(slot))%1000000>=keep_ppm_){++count_["budget_suppressed_addresses"];continue;}
    ++retained;
    if(outputs_.size()>=output_capacity_){++count_["output_dropped_addresses"];continue;}
    outputs_.push_back(Output{active_event_.pc,active_event_.line,address,category,p,static_cast<unsigned>(slot),current,duplicate});
  }
}
void Dynamic602::service(){
  if(!forward_)return;
  if(warmup_complete[cache_->cpu]&&!roi_started_)start_roi();
  if(!roi_started_&&!observing_warmup())return;
  for(;;){
    if(active_&&now()>=due_){
      auto it=entries_.find(active_event_.pc);if(it==entries_.end()||it->second.life!=active_event_.life)throw std::runtime_error("state lifetime changed during inference");
      if(!failed_){it->second.state.hidden=result_.hidden;it->second.state.cell=result_.cell;++it->second.observed_updates;++count_["state_history_completed"];++pc_history_[active_event_.pc].first;count_["state_history_prior_sum"]+=active_event_.prior_updates;peak("state_history_prior_max",active_event_.prior_updates);if(!active_event_.prior_updates)++count_["state_history_zero_prior"];sample("prior_completed_updates",active_event_.prior_updates);append_result();}
      ++count_["nn_completed"];peak("peak_output_queue",outputs_.size());
      count_["inference_completion_cycles_sum"]+=now()-active_event_.arrival_cycle;peak("inference_completion_cycles_max",now()-active_event_.arrival_cycle);sample("completion_age_cycles",now()-active_event_.arrival_cycle);
      count_["inference_completed_service_cycles_sum"]+=now()-active_event_.start_cycle;peak("inference_completed_service_cycles_max",now()-active_event_.start_cycle);sample("completed_service_cycles",now()-active_event_.start_cycle);
      if(roi_started_)log_completion(it->second);active_=false;active_event_=Event();result_=stride_lstm::InferenceResult();
    }
    if(!outputs_.empty()&&(!macs_||now()>=output_next_)){
      const Output item=outputs_.front();outputs_.pop_front();++count_["cache_request_attempts"];++count_[std::string(categories[item.category])+"_attempted"];
      dynamic602_label_request(cache_,item.pc,item.line<<LOG2_BLOCK_SIZE,item.address,item.phase,item.slot,item.current_line,item.list_duplicate,item.category);
      const bool accepted=cache_->prefetch_line(item.pc,item.line<<LOG2_BLOCK_SIZE,item.address,FILL_L2,0);
      ++count_[accepted?"cache_request_accepted":"cache_request_rejected"];++count_[std::string(categories[item.category])+(accepted?"_accepted":"_rejected")];
      output_next_=now()+1;if(!macs_)continue;
    }
    if(!active_&&!inputs_.empty()){Event e=std::move(inputs_.front());inputs_.pop_front();begin(std::move(e));if(!macs_)continue;}
    break;
  }
}
void Dynamic602::log_completion(const Entry& entry){
  if(!history_log_.is_open()||count_["history_log_rows"]>=history_log_limit_)return;
  auto values=[&](const char* name,const std::vector<float>& v){history_log_<<",\""<<name<<"\":[";for(size_t i=0;i<v.size();++i){if(i)history_log_<<',';history_log_<<v[i];}history_log_<<']';};
  history_log_<<"{\"completion\":"<<count_["nn_completed"]<<",\"instructions\":"<<dynamic602_retired_instructions(cache_->cpu)<<",\"eligible\":"<<count_["eligible_l2_callbacks"]<<",\"admitted\":"<<count_["nn_admitted"]<<",\"pc\":"<<active_event_.pc<<",\"line\":"<<active_event_.line<<",\"life\":"<<active_event_.life<<",\"arrival_cycle\":"<<active_event_.arrival_cycle<<",\"start_cycle\":"<<active_event_.start_cycle<<",\"scheduled_finish_cycle\":"<<due_<<",\"finish_cycle\":"<<now()<<",\"prior_updates\":"<<active_event_.prior_updates<<",\"post_updates\":"<<entry.observed_updates<<",\"failed\":"<<(failed_?"true":"false")<<",\"emit\":"<<result_.emit<<",\"K\":"<<result_.k<<",\"teacher_active_at_arrival\":"<<(active_event_.teacher_active?"true":"false");
  values("old_h",active_event_.prior.hidden);values("old_c",active_event_.prior.cell);values("new_h",entry.state.hidden);values("new_c",entry.state.cell);
  history_log_<<",\"teacher_addresses_at_arrival\":[";for(size_t i=0;i<active_event_.teacher_addresses.size();++i){if(i)history_log_<<',';history_log_<<active_event_.teacher_addresses[i];}history_log_<<"],\"candidate_addresses\":[";for(size_t i=0;i<result_.addresses.size();++i){if(i)history_log_<<',';history_log_<<result_.addresses[i];}history_log_<<"]}\n";history_log_.flush();++count_["history_log_rows"];
}
std::string Dynamic602::json()const{
  std::ostringstream s;bool first=true;
  auto add=[&](const std::string& key,uint64_t value){if(!first)s<<',';first=false;s<<'"'<<key<<"\":"<<value;};
  for(const auto& item:count_)add(item.first,item.second);
  add("input_queue",inputs_.size());add("output_queue",outputs_.size());add("inference_inflight",active_);add("state_entries",entries_.size());
  const uint64_t h=forward_?forward_->model().hidden_size():0,p=forward_?forward_->model().parameter_count():0;
  add("weight_bytes",p*4);add("nn_weight_dram_bytes",0);add("dense_weight_copies",forward_?1:0);add("h",h);add("macs_per_cycle",macs_);add("weight_bandwidth_bytes_per_cycle",bandwidth_);add("nonlinear_cycles_per_operation",nonlinear_cycles_);add("nonlinear_units",macs_?1:0);add("state_capacity",capacity_);add("input_queue_capacity",input_capacity_);add("output_queue_capacity",output_capacity_);add("history_instructions",history_instructions_);add("warmup_instructions",warmup_instructions_);add("keep_ppm",keep_ppm_);add("max_requests_per_event",max_requests_per_event_);add("silent_suppress",silent_suppress_);
  add("state_entry_bytes",h?32+8*h:0);add("state_bytes_peak",count_["peak_state_entries"]*(32+8*h));add("state_bytes_configured",capacity_*(32+8*h));add("stride_bytes_configured",stride_?64*32+16:0);add("inference_scratch_bytes",h?(128+26*h)*4:0);add("active_event_metadata_bytes",h?24:0);add("decoded_result_bytes_configured",h?16*(output_limit_?output_limit_:32):0);add("decoded_result_bytes_peak",16*count_["peak_decoded_addresses"]);add("input_queue_bytes_configured",h?input_capacity_*24:0);add("output_queue_bytes_configured",h?output_capacity_*24:0);add("input_queue_bytes_peak",count_["peak_input_queue"]*24);add("output_queue_bytes_peak",count_["peak_output_queue"]*24);
  add("deployment_bytes_configured",h&&capacity_?p*4+capacity_*(32+8*h)+(128+26*h)*4+24+16*(output_limit_?output_limit_:32)+input_capacity_*24+output_capacity_*24:0);
  add("configured_state_unbounded",h&&!capacity_);
  s<<",\"service_architecture\":\"serial_II_equals_service\",\"queue_policy\":\""<<queue_policy_<<"\",\"page_policy\":\""<<page_policy_<<"\",\"stride_mode\":\""<<stride_mode_<<"\",\"min_emit_margin\":"<<min_emit_margin_<<",\"measurement_shadow_bytes\":"<<(2*(64*32+16));
  s<<",\"engine_method\":\""<<method_<<"\",\"engine_payload_mode\":\""<<payload_mode_<<"\",\"mechanism_phase_instructions\":"<<phase_instructions_<<",\"mechanism_phase_count\":"<<ceildiv(roi_instructions_,phase_instructions_);
  add("pc_gate_entries",gate_table_.size());add("pc_gate_capacity",method_=="pc_gate_plus2"?64:0);add("pc_gate_table_packed_bits",method_=="pc_gate_plus2"?64*66:0);
  s<<",\"mechanism_distributions_exact\":{";bool mfirst=true;for(const auto& distribution:mechanism_distributions_){if(!mfirst)s<<',';mfirst=false;s<<'\"'<<distribution.first<<"\":{";bool bucket_first=true;for(const auto& bucket:distribution.second){if(!bucket_first)s<<',';bucket_first=false;s<<'\"'<<bucket.first<<"\":"<<bucket.second;}s<<'}';}s<<'}';
  s<<",\"mechanism_request_profile\":{";mfirst=true;for(const auto& request:mechanism_requests_){if(!mfirst)s<<',';mfirst=false;s<<'\"'<<request.first<<"\":"<<request.second;}s<<'}';
  s<<",\"distributions_log2_upper_bound\":{";bool a=true;for(const auto& distribution:hist_){if(!a)s<<',';a=false;s<<'"'<<distribution.first<<"\":{";bool b=true;for(const auto& bucket:distribution.second){if(!b)s<<',';b=false;s<<'"'<<bucket.first<<"\":"<<bucket.second;}s<<'}';}s<<'}';
  s<<",\"per_pc_roi_updates\":{";a=true;for(const auto& entry:pc_history_){if(!a)s<<',';a=false;s<<'"'<<entry.first<<"\":{\"completed\":"<<entry.second.first<<",\"evictions\":"<<entry.second.second<<'}';}s<<'}';
  s<<",\"pc_completed_updates\":{";a=true;for(const auto& entry:pc_history_){if(!a)s<<',';a=false;s<<'"'<<entry.first<<"\":"<<entry.second.first;}s<<'}';
  s<<",\"pc_state_evictions\":{";a=true;for(const auto& entry:pc_history_){if(!a)s<<',';a=false;s<<'"'<<entry.first<<"\":"<<entry.second.second;}s<<'}';
  s<<",\"pc_warmup_updates\":{";a=true;for(const auto& entry:warmup_pc_history_){if(!a)s<<',';a=false;s<<'"'<<entry.first<<"\":{\"completed\":"<<entry.second.first<<",\"evictions\":"<<entry.second.second<<'}';}s<<'}';
  s<<",\"resident_pc_lifetime_updates\":{";a=true;for(const auto& entry:entries_){if(!a)s<<',';a=false;s<<'"'<<entry.first<<"\":{\"life\":"<<entry.second.life<<",\"completed_updates\":"<<entry.second.observed_updates<<'}';}s<<'}';return s.str();
}
void Dynamic602::print_config(){std::cout<<"dynamic602 method "<<method_<<" macs "<<macs_<<" state_capacity "<<capacity_<<" output_limit "<<output_limit_<<std::endl;std::cout<<"dynamic602_protocol {"<<json()<<"}"<<std::endl;}
void Dynamic602::dump_stats(){if(input_log_)input_log_.flush();std::cout<<"dynamic602_final {"<<json()<<"}"<<std::endl;}
void dynamic602_service(CACHE* cache){auto it=engines.find(cache->cpu);if(it!=engines.end())it->second->service();}
void dynamic602_roi(CACHE* cache){auto it=engines.find(cache->cpu);if(it!=engines.end())it->second->start_roi();}
std::string dynamic602_engine_json(unsigned cpu){auto it=engines.find(cpu);return it==engines.end()?"":it->second->json();}
