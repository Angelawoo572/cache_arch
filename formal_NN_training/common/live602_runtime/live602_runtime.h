#ifndef LIVE602_RUNTIME_H
#define LIVE602_RUNTIME_H
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace live602 {
struct State { std::vector<float> hidden,cell; };
struct Action { uint64_t address=0;int64_t delta=0;int fill=2,fill_index=-1;float coordinate=0; };
struct Result {
  bool demand=true,emit=false,truncated=false;
  uint64_t count=0,macs=0,nonlinears=0;
  float count_raw=0;
  std::vector<float> hidden,cell,emit_logits;
  std::vector<Action> actions;
};
struct Config {
  uint32_t version=1,endian=0x01020304,float_type=1,tensor_count=0;
  uint32_t hidden_size=0,feature_count=0,mixture_components=0,fill_classes=0;
  uint32_t count_mode=0,decoder_initial=0,kind_feature=0,demand_kind=0;
  uint32_t delta_limit_bits=57,fill_level0=2,fill_level1=4,delta_mode=0;
  uint64_t parameter_count=0;
};
struct Tensor { std::vector<uint32_t> shape;std::vector<float> data; };
class Live602Runtime {
 public:
  explicit Live602Runtime(const std::string& path);
  void Reset();
  const State& GetState()const{return state_;}
  void SetState(const State& state);
  const Config& config()const{return config_;}
  uint64_t weight_bytes()const{return config_.parameter_count*4;}
  Result Infer(uint64_t aligned_byte_address,bool demand=true);
  Result InferState(uint64_t aligned_byte_address,bool demand,const State& state)const;
 private:
  Config config_;State state_;std::map<std::string,Tensor> tensors_;
  const Tensor& tensor(const std::string& name)const;
  std::vector<float> linear(const std::string& name,const std::vector<float>& input,Result& result)const;
  void load(const std::string& path);
  void check_state(const State& state)const;
};
}
#endif
