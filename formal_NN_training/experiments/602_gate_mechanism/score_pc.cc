// Full-stream scalar NN scorer for a fixed-capacity, offline PC-gate baseline.
// This is not a ChampSim performance run and has no maximum-event shortcut.
#include "stride_lstm_runtime.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

struct State { stride_lstm::RecurrentState value; uint64_t lru=0; };
struct Votes { uint64_t count=0, positives=0, emit_k_zero=0; };
static uint64_t number(const std::string& s) {
  if(s.empty()||s[0]=='-')throw std::runtime_error("invalid unsigned integer");
  size_t n=0;auto x=std::stoull(s,&n,10);
  if(n!=s.size())throw std::runtime_error("invalid unsigned integer");
  return x;
}
int main(int argc,char** argv) {
  try {
    if(argc!=5)throw std::runtime_error("usage: score_pc MODEL CANONICAL_PC_LINE_CSV COUNTS_CSV SUMMARY_JSON");
    stride_lstm::StrideLSTMRuntime runtime(argv[1]);runtime.SetOutputLimit(32);
    std::ifstream input(argv[2]);std::ofstream output(argv[3]),summary(argv[4]);
    if(!input||!output||!summary)throw std::runtime_error("cannot open scorer file");
    std::string row;if(!std::getline(input,row)||row!="pc,line")throw std::runtime_error("expected canonical pc,line CSV");
    std::map<uint64_t,State> states;std::map<uint64_t,Votes> votes;
    uint64_t events=0,evictions=0,clock=0,positives=0;
    while(std::getline(input,row)) {
      if(!row.empty()&&row.back()=='\r')row.pop_back();
      if(row.empty())throw std::runtime_error("empty scorer event");
      auto comma=row.find(',');
      if(comma==std::string::npos||row.find(',',comma+1)!=std::string::npos)throw std::runtime_error("malformed canonical event");
      uint64_t pc=number(row.substr(0,comma)),line=number(row.substr(comma+1));
      auto it=states.find(pc);
      if(it==states.end()) {
        if(states.size()>=64) {
          auto victim=std::min_element(states.begin(),states.end(),[](const auto& a,const auto& b){return a.second.lru<b.second.lru;});
          states.erase(victim);++evictions;
        }
        State fresh;fresh.value.hidden.assign(runtime.model().hidden_size(),0);fresh.value.cell=fresh.value.hidden;
        it=states.emplace(pc,std::move(fresh)).first;
      }
      it->second.lru=++clock;
      auto result=runtime.InferState(pc,line,it->second.value);
      it->second.value.hidden=std::move(result.hidden);it->second.value.cell=std::move(result.cell);
      auto& v=votes[pc];++v.count;
      if(result.emit){++v.positives;++positives;if(result.k==0)++v.emit_k_zero;}
      ++events;
    }
    if(!events)throw std::runtime_error("empty scorer stream");
    output<<"pc,count,positives,emit_k_zero\n";
    for(const auto& pair:votes)output<<pair.first<<','<<pair.second.count<<','<<pair.second.positives<<','<<pair.second.emit_k_zero<<'\n';
    summary<<"{\"events\":"<<events<<",\"positive_emits\":"<<positives
      <<",\"unique_pcs\":"<<votes.size()<<",\"state_capacity\":64,\"state_evictions\":"<<evictions
      <<",\"peak_state_entries\":"<<std::min<size_t>(64,votes.size())
      <<",\"split_initial_state\":\"zero\",\"full_stream\":true,\"output_limit\":32}\n";
    std::cout<<"SCORED_FULL_STREAM events="<<events<<" unique_pcs="<<votes.size()<<'\n';
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
