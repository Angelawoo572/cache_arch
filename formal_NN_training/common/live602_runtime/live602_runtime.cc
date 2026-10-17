#include "live602_runtime.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
namespace live602 {
namespace {
uint64_t integer(std::istream& f,unsigned n){uint64_t value=0;for(unsigned i=0;i<n;++i){int c=f.get();if(c<0)throw std::runtime_error("truncated LV602R1 model");value|=uint64_t(c)<<(8*i);}return value;}
float scalar(std::istream& f){uint32_t bits=integer(f,4);float value;std::memcpy(&value,&bits,4);if(!std::isfinite(value))throw std::runtime_error("nonfinite model parameter");return value;}
float sigmoid(float value){if(value>=0){const float t=std::exp(-value);return 1.0F/(1.0F+t);}const float t=std::exp(value);return t/(1.0F+t);}
float softplus(float value){return value>20.0F?value:std::log1p(std::exp(value));}
float dot(const Tensor& matrix,uint32_t row,const std::vector<float>& input){float result=0;const size_t width=matrix.shape.at(1);if(width!=input.size())throw std::runtime_error("live602 dot shape mismatch");for(size_t j=0;j<width;++j)result+=matrix.data[row*width+j]*input[j];return result;}
size_t argmax(const std::vector<float>& values,size_t begin,size_t count){if(!count)throw std::runtime_error("empty live602 argmax");size_t best=begin;for(size_t i=begin+1;i<begin+count;++i)if(values[i]>values[best])best=i;return best-begin;}
const uint64_t line_mask=(1ULL<<58)-1;
}
Live602Runtime::Live602Runtime(const std::string& path){static_assert(sizeof(float)==4,"FP32 required");load(path);Reset();}
void Live602Runtime::Reset(){state_.hidden.assign(config_.hidden_size,0);state_.cell=state_.hidden;}
void Live602Runtime::check_state(const State& state)const{if(state.hidden.size()!=config_.hidden_size||state.cell.size()!=config_.hidden_size)throw std::runtime_error("live602 recurrent state shape mismatch");for(float v:state.hidden)if(!std::isfinite(v))throw std::runtime_error("nonfinite hidden state");for(float v:state.cell)if(!std::isfinite(v))throw std::runtime_error("nonfinite cell state");}
void Live602Runtime::SetState(const State& state){check_state(state);state_=state;}
const Tensor& Live602Runtime::tensor(const std::string& name)const{auto found=tensors_.find(name);if(found==tensors_.end())throw std::runtime_error("missing tensor "+name);return found->second;}
std::vector<float> Live602Runtime::linear(const std::string& name,const std::vector<float>& input,Result& result)const{const Tensor& w=tensor(name+".weight");const Tensor& b=tensor(name+".bias");std::vector<float> output(w.shape.at(0));for(size_t i=0;i<output.size();++i)output[i]=b.data[i]+dot(w,i,input);result.macs+=output.size()*input.size();for(float v:output)if(!std::isfinite(v))throw std::runtime_error("nonfinite affine output");return output;}
void Live602Runtime::load(const std::string& path){
 std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open live602 model");char magic[8];f.read(magic,8);if(f.gcount()!=8||std::memcmp(magic,"LV602R1\0",8))throw std::runtime_error("bad LV602R1 magic");
 uint32_t* fields[]={&config_.version,&config_.endian,&config_.float_type,&config_.tensor_count,&config_.hidden_size,&config_.feature_count,&config_.mixture_components,&config_.fill_classes,&config_.count_mode,&config_.decoder_initial,&config_.kind_feature,&config_.demand_kind,&config_.delta_limit_bits,&config_.fill_level0,&config_.fill_level1,&config_.delta_mode};
 for(auto field:fields)*field=integer(f,4);config_.parameter_count=integer(f,8);
 if(config_.version!=1||config_.endian!=0x01020304||config_.float_type!=1||!config_.hidden_size||config_.feature_count!=64+config_.kind_feature||config_.kind_feature>1||config_.demand_kind>1||config_.count_mode>1||config_.decoder_initial>1||config_.delta_mode>1||!config_.mixture_components||(config_.fill_classes!=0&&config_.fill_classes!=2)||config_.delta_limit_bits>63)throw std::runtime_error("invalid live602 schema");
 uint64_t parameters=0;for(uint32_t i=0;i<config_.tensor_count;++i){const uint64_t namesize=integer(f,2),rank=integer(f,2),count=integer(f,8);if(!namesize||rank>4||!rank)throw std::runtime_error("invalid tensor header");Tensor value;uint64_t product=1;for(uint64_t j=0;j<rank;++j){uint32_t dimension=integer(f,4);if(!dimension||product>std::numeric_limits<uint64_t>::max()/dimension)throw std::runtime_error("invalid tensor dimension");value.shape.push_back(dimension);product*=dimension;}if(product!=count||count>config_.parameter_count)throw std::runtime_error("invalid tensor count");std::string name(namesize,'\0');f.read(&name[0],namesize);if(!f)throw std::runtime_error("truncated tensor name");value.data.reserve(count);for(uint64_t j=0;j<count;++j)value.data.push_back(scalar(f));if(!tensors_.emplace(name,std::move(value)).second)throw std::runtime_error("duplicate tensor");parameters+=count;}
 if(parameters!=config_.parameter_count||f.peek()!=std::char_traits<char>::eof())throw std::runtime_error("live602 model size mismatch");
 const uint32_t h=config_.hidden_size,features=config_.feature_count,inputs=1+config_.fill_classes,delta_rows=config_.delta_mode?1:3*config_.mixture_components;
 std::map<std::string,std::vector<uint32_t>> expected={{"encoder.weight_ih",{4*h,features}},{"encoder.weight_hh",{4*h,h}},{"encoder.bias_ih",{4*h}},{"encoder.bias_hh",{4*h}},{"count.weight",{1,h}},{"count.bias",{1}},{"gru.weight_ih",{3*h,inputs}},{"gru.weight_hh",{3*h,h}},{"gru.bias_ih",{3*h}},{"gru.bias_hh",{3*h}},{"delta.weight",{delta_rows,h}},{"delta.bias",{delta_rows}}};
 if(config_.count_mode){expected["emit.weight"]={2,h};expected["emit.bias"]={2};}if(!config_.decoder_initial){expected["initial.weight"]={h,h};expected["initial.bias"]={h};}if(config_.fill_classes){expected["fill.weight"]={config_.fill_classes,h};expected["fill.bias"]={config_.fill_classes};}
 if(expected.size()!=tensors_.size())throw std::runtime_error("unexpected tensor set");for(const auto& item:expected)if(tensor(item.first).shape!=item.second)throw std::runtime_error("live602 tensor shape mismatch: "+item.first);
}
Result Live602Runtime::Infer(uint64_t address,bool demand){Result result=InferState(address,demand,state_);SetState(State{result.hidden,result.cell});return result;}
Result Live602Runtime::InferState(uint64_t address,bool demand,const State& prior)const{
 check_state(prior);if(address&63)throw std::runtime_error("live602 expects aligned byte address");if(!demand&&!config_.kind_feature)throw std::runtime_error("address-only student cannot observe fill events");
 Result result;result.demand=demand;const uint32_t h=config_.hidden_size;std::vector<float> features(config_.feature_count,0);for(unsigned bit=0;bit<64;++bit)features[bit]=(address>>bit)&1ULL;if(config_.kind_feature)features[64]=demand?config_.demand_kind:1-config_.demand_kind;
 const Tensor& wi=tensor("encoder.weight_ih");const Tensor& wh=tensor("encoder.weight_hh");const Tensor& bi=tensor("encoder.bias_ih");const Tensor& bh=tensor("encoder.bias_hh");std::vector<float> gates(4*h);
 for(uint32_t i=0;i<4*h;++i)gates[i]=bi.data[i]+bh.data[i]+dot(wi,i,features)+dot(wh,i,prior.hidden);
 result.macs+=4*h*(config_.feature_count+h);result.nonlinears+=5*h;result.hidden.resize(h);result.cell.resize(h);
 for(uint32_t i=0;i<h;++i){const float ig=sigmoid(gates[i]),fg=sigmoid(gates[h+i]),candidate=std::tanh(gates[2*h+i]),og=sigmoid(gates[3*h+i]);result.cell[i]=fg*prior.cell[i]+ig*candidate;result.hidden[i]=og*std::tanh(result.cell[i]);}
 check_state(State{result.hidden,result.cell});if(!demand)return result;
 bool emit=true;if(config_.count_mode){result.emit_logits=linear("emit",result.hidden,result);emit=result.emit_logits[1]>result.emit_logits[0];}
 if(emit){result.count_raw=linear("count",result.hidden,result)[0];double count;
  if(config_.count_mode){count=std::nearbyint(std::exp(static_cast<double>(result.count_raw)));++result.nonlinears;}
  else {const double raw=result.count_raw;const double rate=std::max(raw,0.0)+std::log1p(std::exp(-std::fabs(raw)));count=std::floor(rate);++result.nonlinears;}
  if(!std::isfinite(count)||count>=9223372036854775808.0||count<0)throw std::runtime_error("live602 count outside int64 domain");result.count=static_cast<uint64_t>(count);if(config_.count_mode&&result.count<1)result.count=1;
 }
 result.emit=result.count!=0;if(!result.count)return result;
 std::vector<float> decoder=result.hidden;if(!config_.decoder_initial){decoder=linear("initial",result.hidden,result);for(float& value:decoder)value=std::tanh(value);result.nonlinears+=h;}
 const Tensor& gwi=tensor("gru.weight_ih");const Tensor& gwh=tensor("gru.weight_hh");const Tensor& gbi=tensor("gru.bias_ih");const Tensor& gbh=tensor("gru.bias_hh");
 for(uint64_t slot=0;slot<result.count;++slot){
  std::vector<float> raw=linear("delta",decoder,result);float coordinate;
  if(config_.delta_mode)coordinate=raw[0];else {const uint32_t mixtures=config_.mixture_components;size_t selected=argmax(raw,0,mixtures);coordinate=raw[mixtures+selected];for(uint32_t j=0;j<mixtures;++j){volatile float scale=softplus(raw[2*mixtures+j])+std::numeric_limits<float>::min();(void)scale;}result.nonlinears+=mixtures;}
  if(!std::isfinite(coordinate))throw std::runtime_error("nonfinite live602 delta");const double magnitude=std::expm1(std::fabs(static_cast<double>(coordinate)));++result.nonlinears;const double limit=std::ldexp(1.0,config_.delta_limit_bits);if(!std::isfinite(magnitude)||magnitude>limit||magnitude>=9223372036854775808.0)throw std::runtime_error("live602 delta outside address domain");int64_t delta=static_cast<int64_t>(std::nearbyint(magnitude));if(coordinate<0)delta=-delta;
  Action action;action.delta=delta;action.address=(((address>>6)+static_cast<uint64_t>(delta))&line_mask)<<6;action.coordinate=coordinate;action.fill=config_.fill_level0;
  if(config_.fill_classes){const std::vector<float> fill=linear("fill",decoder,result);action.fill_index=argmax(fill,0,fill.size());action.fill=action.fill_index?config_.fill_level1:config_.fill_level0;}
  result.actions.push_back(action);std::vector<float> input(1+config_.fill_classes,0);input[0]=coordinate;if(config_.fill_classes)input[1+action.fill_index]=1;
  std::vector<float> next(h);for(uint32_t i=0;i<h;++i){const float ir=gbi.data[i]+dot(gwi,i,input),iz=gbi.data[h+i]+dot(gwi,h+i,input),in=gbi.data[2*h+i]+dot(gwi,2*h+i,input);const float hr=gbh.data[i]+dot(gwh,i,decoder),hz=gbh.data[h+i]+dot(gwh,h+i,decoder),hn=gbh.data[2*h+i]+dot(gwh,2*h+i,decoder);const float reset=sigmoid(ir+hr),update=sigmoid(iz+hz),candidate=std::tanh(in+reset*hn);next[i]=(1-update)*candidate+update*decoder[i];}
  result.macs+=3*h*(input.size()+h);result.nonlinears+=3*h;decoder.swap(next);
 }
 return result;
}
}
