#include "online602_engine.h"
#include "cache.h"
#include "champsim.h"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {
std::map<unsigned,Online602*> engines;
uint64_t envnum(const char* key,uint64_t fallback) {const char* p=std::getenv(key);return p?std::stoull(p):fallback;}
std::string envstr(const char* key,const char* fallback) {const char* p=std::getenv(key);return p?p:fallback;}
void bytes(int fd,void* data,size_t size,bool send) {
  char* p=static_cast<char*>(data);
  while(size) {ssize_t n=send?::write(fd,p,size) : ::read(fd,p,size);
    if(n<0 && errno==EINTR)continue;
    if(n<=0)throw std::runtime_error("online training IPC closed/failed");
    p+=n;size-=n;
  }
}
template<class T> T get(const std::vector<char>& b,size_t at) {T x; if(at+sizeof(T)>b.size())throw std::runtime_error("short worker response");std::memcpy(&x,b.data()+at,sizeof(T));return x;}
uint64_t ceildiv(uint64_t n,uint64_t d){return (n+d-1)/d;}
}

Online602::Online602(std::string type,CACHE* cache):Prefetcher(type),cache_(cache) {
  method_=envstr("ONLINE602_METHOD","frozen");
  if(method_!="none"&&method_!="stride"&&method_!="frozen"&&method_!="online")throw std::runtime_error("unknown online602 method");
  macs_=envnum("ONLINE602_MACS",0);capacity_=envnum("ONLINE602_STATE_CAPACITY",0);
  output_limit_=envnum("ONLINE602_OUTPUT_LIMIT",32);
  if(macs_!=0 && macs_!=4 && macs_!=16)throw std::runtime_error("service case must be 0,4,16");
  if(method_=="stride"||method_=="online")teacher_.reset(new StridePrefetcher("stride"));
  if(method_=="frozen"||method_=="online") {
    forward_.reset(new stride_lstm::StrideLSTMRuntime(envstr("STRIDE_LSTM_MODEL_BIN","")));
    forward_->SetOutputLimit(output_limit_);
  }
  if(method_=="online") {
    socket_=::socket(AF_UNIX,SOCK_STREAM,0);if(socket_<0)throw std::runtime_error("cannot create training socket");
    sockaddr_un address;std::memset(&address,0,sizeof(address));address.sun_family=AF_UNIX;
    std::string path=envstr("ONLINE602_SOCKET","");
    if(path.size()>=sizeof(address.sun_path))throw std::runtime_error("socket path too long");
    std::strcpy(address.sun_path,path.c_str());
    if(::connect(socket_,reinterpret_cast<sockaddr*>(&address),sizeof(address)))throw std::runtime_error("cannot connect training worker");
  }
  for(const char* key:{"eligible_l2_callbacks","nn_admitted","nn_completed","inference_dropped","numerical_failures","decoder_dropped_addresses","output_dropped_addresses","available_supervised_decisions","available_positive_decisions","available_positive_actions","training_admitted","training_dropped","completed_updates","sample_exposures","published_versions","state_evictions","peak_state_entries","peak_input_queue","peak_output_queue","peak_training_queue","inference_macs","inference_nonlinears","training_forward_macs","training_modeled_macs","training_modeled_nonlinears","inference_service_cycles","training_service_cycles","publication_cycles","teacher_calls","teacher_tag_comparison_bound","max_tbptt_span","peak_padded_positions","peak_inflight_updates","published_parameter_values","prediction_weight_version_max"})count_[key]=0;
  engines[cache_->cpu]=this;print_config();
}
Online602::~Online602(){if(socket_>=0)::close(socket_);engines.erase(cache_->cpu);}
uint64_t Online602::now()const{return current_core_cycle[cache_->cpu];}
void Online602::peak(const std::string& name,uint64_t value){count_[name]=std::max(count_[name],value);}
uint64_t Online602::latency(uint64_t mac,uint64_t nonlinear,uint64_t control)const {
  if(!macs_)return 0;
  // Matched 4*MACs bytes/cycle weights; one nonlinear unit, 4 cycles/op.
  // Dense arithmetic and nonlinear/control phases serialize conservatively.
  return ceildiv(mac,macs_)+4*nonlinear+control;
}
void Online602::invoke_prefetcher(uint64_t pc,uint64_t address,uint8_t hit,uint8_t access,std::vector<uint64_t>& output){
  if(access!=LOAD||!warmup_complete[cache_->cpu])return;
  ++count_["eligible_l2_callbacks"];
  if(method_=="none")return;
  if(method_=="stride"){teacher_->invoke_prefetcher(pc,address,hit,access,output);return;}
  service();
  Event event;event.pc=pc;event.line=address>>LOG2_BLOCK_SIZE;event.label_ready=now();
  if(teacher_){
    std::vector<uint64_t> labels;
    teacher_->invoke_prefetcher(pc,address,hit,access,labels);
    for(auto label:labels)event.actions.push_back(label>>LOG2_BLOCK_SIZE);
    ++count_["teacher_calls"];count_["teacher_tag_comparison_bound"]+=64;
    ++count_["observed_teacher_decisions"];
    count_["observed_teacher_positive_decisions"]+=!labels.empty();
    count_["observed_teacher_actions"]+=labels.size();
    event.label_ready+=macs_?24:0;
    if(labels_.size()<64){event.supervised=true;labels_.push_back(Label{event.label_ready,uint64_t(!labels.empty()),labels.size()});peak("peak_teacher_queue",labels_.size());}
    else ++count_["teacher_label_dropped"];
  }
  if(inputs_.size()>=16){++count_["inference_dropped"];return;}
  inputs_.push_back(std::move(event));++count_["nn_admitted"];
  peak("peak_input_queue",inputs_.size());service();
}
void Online602::begin(Event event){
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
  count_["prediction_weight_version_max"]=version_;
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
void Online602::train(){
  if(method_!="online"||update_||training_.size()<64)return;
  // Chronological queue, never await future events to complete a PC sequence.
  if(training_[63].label_ready>now())return;
  std::ostringstream s;s<<std::setprecision(9)<<"{\"op\":\"train\",\"events\":[";
  std::map<uint64_t,uint64_t> lengths;uint64_t positives=0,actions=0;
  for(size_t i=0;i<64;++i){const Event& e=training_[i];if(i)s<<',';
    s<<"{\"pc\":"<<e.pc<<",\"line\":"<<e.line<<",\"life\":"<<e.life<<",\"h\":[";
    for(size_t j=0;j<e.prior.hidden.size();++j){if(j)s<<',';s<<e.prior.hidden[j];}
    s<<"],\"c\":[";for(size_t j=0;j<e.prior.cell.size();++j){if(j)s<<',';s<<e.prior.cell[j];}
    s<<"],\"actions\":[";for(size_t j=0;j<e.actions.size();++j){if(j)s<<',';s<<e.actions[j];}s<<"]}";
    ++lengths[e.life];positives+=!e.actions.empty();actions+=e.actions.size();
  }s<<"]}";
  std::string request=s.str();uint32_t size=request.size();bytes(socket_,&size,sizeof(size),true);bytes(socket_,&request[0],request.size(),true);
  training_n_=64;for(size_t i=0;i<64;++i)training_.pop_front();
  ++count_["updates_started"];
  uint64_t span=0;for(const auto& item:lengths)span=std::max(span,item.second);
  const uint64_t padded=span*lengths.size(),h=forward_->model().hidden_size(),p=forward_->model().parameter_count();
  const uint64_t f=padded*128*h+64*(8*h*h+2*h)+positives*h+actions*(3*h*h+4*h);
  const uint64_t nonlinear=padded*h+64*5*h+actions*3*h+64*3+positives+2*p;
  // Costed backward upper bound: at most two dense MACs per forward MAC.
  const uint64_t work=3*f;
  const uint64_t traffic=32*p+4*(padded*128+128*h+actions*h);
  count_["training_state_traffic_bytes"]+=traffic;
  const uint64_t cycles=latency(work,3*nonlinear,64*8+12*p)+(macs_?ceildiv(traffic,4*macs_):0);
  count_["training_forward_macs"]+=f;count_["training_modeled_macs"]+=work;count_["training_modeled_nonlinears"]+=3*nonlinear;
  count_["training_service_cycles"]+=cycles;
  peak("max_tbptt_span",span);peak("peak_padded_positions",padded);
  update_=true;received_=false;peak("peak_inflight_updates",1);train_due_=now()+cycles;
  const uint64_t copy=macs_?(ceildiv(forward_->model().weight_bytes(),4*macs_)+8):0;
  count_["publication_cycles"]+=copy;publish_due_=train_due_+copy;
}
void Online602::receive(){
  uint32_t header[2];bytes(socket_,header,sizeof(header),false);
  if(header[1]>1024*1024)throw std::runtime_error("oversized worker response");
  std::vector<char> payload(header[1]);bytes(socket_,payload.data(),payload.size(),false);
  if(header[0])throw std::runtime_error(std::string(payload.begin(),payload.end()));
  const size_t parameters=forward_->model().parameter_count();
  if(payload.size()!=56+4*parameters)throw std::runtime_error("worker weight response shape mismatch");
  const uint64_t response_version=get<uint64_t>(payload,0);
  if(response_version!=version_+1)throw std::runtime_error("worker publication version mismatch");
  publication_.resize(parameters);std::memcpy(publication_.data(),payload.data()+56,4*parameters);
  count_["sample_exposures"]+=training_n_;++count_["completed_updates"];received_=true;
}
void Online602::service(){
  if(!warmup_complete[cache_->cpu]||!forward_)return;
  while(!labels_.empty() && labels_.front().ready<=now()){
    const Label label=labels_.front();labels_.pop_front();
    ++count_["available_supervised_decisions"];count_["available_positive_decisions"]+=label.positive;count_["available_positive_actions"]+=label.actions;
  }
  // Host IPC may wait here for CPU co-simulation, but adds NO simulated cycles.
  if(update_&&!received_&&now()>=train_due_)receive();
  if(update_&&received_&&now()>=publish_due_){
    if(active_&&now()<due_){++count_["publications_during_inference"];peak("retained_weight_bank_bytes_peak",forward_->model().weight_bytes());}
    forward_->Publish(publication_);publication_.clear();update_=false;received_=false;++version_;++count_["published_versions"];count_["published_parameter_values"]+=forward_->model().parameter_count();}
  // A serial engine respects all recurrent dependencies; h/c become visible only here.
  for(;;){
    if(active_&&now()>=due_){
      auto it=entries_.find(active_event_.pc);
      if(it==entries_.end()||it->second.life!=active_event_.life)throw std::runtime_error("state lifetime changed during inference");
      if(!failed_){it->second.state.hidden=result_.hidden;it->second.state.cell=result_.cell;
        for(auto address:result_.addresses){if(outputs_.size()>=32){++count_["output_dropped_addresses"];continue;}outputs_.push_back(Output{active_event_.pc,active_event_.line,address});}}
      ++count_["nn_completed"];peak("peak_output_queue",outputs_.size());
      if(method_=="online" && active_event_.supervised){
        if(training_.size()<64){training_.push_back(std::move(active_event_));++count_["training_admitted"];peak("peak_training_queue",training_.size());}
        else ++count_["training_dropped"];
      }
      active_=false;result_=stride_lstm::InferenceResult();train();
      if(update_&&!macs_){receive();forward_->Publish(publication_);publication_.clear();update_=false;received_=false;++version_;++count_["published_versions"];count_["published_parameter_values"]+=forward_->model().parameter_count();}
    }
    if(!outputs_.empty() && (!macs_ || now()>=output_next_)){
      const Output item=outputs_.front();outputs_.pop_front();
      cache_->prefetch_line(item.pc,item.line<<LOG2_BLOCK_SIZE,item.address,FILL_L2,0);
      output_next_=now()+1;
      if(!macs_)continue;
    }
    train();
    if(!active_&&!inputs_.empty()){Event e=std::move(inputs_.front());inputs_.pop_front();begin(std::move(e));if(!macs_)continue;}
    break;
  }
}
std::string Online602::json()const{
  std::ostringstream s;bool first=true;
  auto add=[&](const std::string& key,uint64_t value){if(!first)s<<',';first=false;s<<'"'<<key<<"\":"<<value;};
  for(const auto& item:count_)add(item.first,item.second);
  add("model_version",version_);add("input_queue",inputs_.size());add("output_queue",outputs_.size());add("training_queue",training_.size());add("inference_inflight",active_);add("update_inflight",update_);add("state_entries",entries_.size());
  const uint64_t h=forward_?forward_->model().hidden_size():0,p=forward_?forward_->model().parameter_count():0;
  add("weight_bytes",p*4);add("h",h);add("macs_per_cycle",macs_);add("weight_bandwidth_bytes_per_cycle",4*macs_);add("state_capacity",capacity_);
  add("state_entry_bytes",h?32+8*h:0);add("state_bytes_peak",count_["peak_state_entries"]*(32+8*h));
  add("state_bytes_configured",capacity_*(32+8*h));add("teacher_bytes_configured",teacher_?64*32+16:0);
  add("teacher_output_bytes_configured",method_=="online"?64*48:0);add("teacher_output_bytes_peak",count_["peak_teacher_queue"]*48);
  add("inference_scratch_bytes",h?(128+26*h)*4:0);
  // Scratch reserves all temporary and carried inference FP32 vectors, including
  // active pre-state and staged result h/c; separately account integer payloads.
  add("active_event_metadata_bytes",h?56:0);
  add("decoded_result_bytes_configured",h?16*(output_limit_?output_limit_:32):0);
  add("decoded_result_bytes_peak",16*count_["peak_decoded_addresses"]);
  add("retained_weight_bank_bytes_configured",method_=="online"&&macs_?4*p:0);
  add("input_queue_bytes_configured",h?16*56:0);add("output_queue_bytes_configured",h?32*24:0);
  add("input_queue_bytes_peak",count_["peak_input_queue"]*56);add("output_queue_bytes_peak",count_["peak_output_queue"]*24);
  add("training_examples_bytes_configured",method_=="online"?128*(56+8*h):0);
  add("publication_bytes_configured",method_=="online"?p*4:0);
  return s.str();
}
void Online602::print_config(){std::cout<<"online602 method "<<method_<<" macs "<<macs_<<" state_capacity "<<capacity_<<" output_limit "<<output_limit_<<std::endl;}
void Online602::dump_stats(){std::cout<<"online602_final {"<<json()<<"}"<<std::endl;}
void online602_service(CACHE* cache){auto it=engines.find(cache->cpu);if(it!=engines.end())it->second->service();}
std::string online602_engine_json(unsigned cpu){auto it=engines.find(cpu);return it==engines.end()?"":it->second->json();}
