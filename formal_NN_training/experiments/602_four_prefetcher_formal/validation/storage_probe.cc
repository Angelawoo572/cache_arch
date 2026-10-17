// Host requested-allocation instrumentation, not a hardware storage/area model.
#include "live602_runtime.h"
#include "streamer.h"
#include "ampm.h"
#include "spp_dev2.h"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
namespace allocation {
struct alignas(std::max_align_t) Header {std::size_t bytes;bool tracked;};
bool enabled=false;std::size_t live=0,peak=0,calls=0,frees=0;
void* acquire(std::size_t n){
 if(n>std::numeric_limits<std::size_t>::max()-sizeof(Header))throw std::bad_alloc();
 Header* p=static_cast<Header*>(std::malloc(sizeof(Header)+std::max<std::size_t>(n,1)));if(!p)throw std::bad_alloc();
 p->bytes=n;p->tracked=enabled;if(enabled){live+=n;peak=std::max(peak,live);++calls;}return p+1;
}
void release(void* value)noexcept{if(!value)return;Header* p=static_cast<Header*>(value)-1;if(p->tracked){live-=p->bytes;++frees;}std::free(p);}
}
void* operator new(std::size_t n){return allocation::acquire(n);}
void* operator new[](std::size_t n){return allocation::acquire(n);}
void operator delete(void* p)noexcept{allocation::release(p);}
void operator delete[](void* p)noexcept{allocation::release(p);}
#if __cplusplus >= 201402L
void operator delete(void* p,std::size_t)noexcept{allocation::release(p);}
void operator delete[](void* p,std::size_t)noexcept{allocation::release(p);}
#endif
int main(int argc,char** argv){try{
 if(argc!=2)throw std::runtime_error("usage: storage_probe MODEL; stdin D/F unsigned_decimal_address");
 const std::string path=argv[1];
 // Parse input outside the accounting window. Input fixture is diagnostic data.
 std::vector<std::pair<uint64_t,bool>> inputs;char kind;uint64_t address;
 while(std::cin>>kind>>address){if((kind!='D'&&kind!='F')||(address&63))throw std::runtime_error("bad aligned input");inputs.emplace_back(address,kind=='D');}
 if(!std::cin.eof()||inputs.empty())throw std::runtime_error("missing or malformed fixture");
 alignas(live602::Live602Runtime) unsigned char storage[sizeof(live602::Live602Runtime)];
 allocation::enabled=true;auto* model=new(storage) live602::Live602Runtime(path);
 const std::size_t persistent=allocation::live,constructor_peak=allocation::peak,constructor_calls=allocation::calls;
 const uint64_t weights=model->weight_bytes(),state_bytes=4*(model->GetState().hidden.size()+model->GetState().cell.size());
 const auto config=model->config();
 std::size_t max_incremental_peak=0,max_returned_live=0,max_actions_capacity=0,inference_allocation_calls=0;
 uint64_t callbacks=0,demands=0,fills=0,actions=0,macs=0,nonlinears=0,max_count=0;
 std::size_t first_peak=0,first_retained=0;uint64_t first_count=0,first_macs=0,first_nonlinears=0;
 for(const auto& input:inputs){
  if(allocation::live!=persistent)throw std::runtime_error("prior inference allocation leak");
  allocation::peak=persistent;const std::size_t calls_before=allocation::calls;
  {auto result=model->Infer(input.first,input.second);++callbacks;if(input.second)++demands;else ++fills;
   const std::size_t extra_peak=allocation::peak-persistent,retained=allocation::live-persistent;
   max_incremental_peak=std::max(max_incremental_peak,extra_peak);max_returned_live=std::max(max_returned_live,retained);
   max_actions_capacity=std::max(max_actions_capacity,result.actions.capacity());max_count=std::max(max_count,result.count);
   actions+=result.actions.size();macs+=result.macs;nonlinears+=result.nonlinears;
   inference_allocation_calls+=allocation::calls-calls_before;
   if(callbacks==1){first_peak=extra_peak;first_retained=retained;first_count=result.count;first_macs=result.macs;first_nonlinears=result.nonlinears;}
  }
 }
 if(allocation::live!=persistent)throw std::runtime_error("final inference allocation leak");
 model->~Live602Runtime();const std::size_t residual=allocation::live;allocation::enabled=false;
 if(residual)throw std::runtime_error("runtime destruction left tracked allocations");
 std::cout<<"{\"schema\":\"live602_host_requested_allocations_v1\",\"sizeof\":{";
#define SIZE(T) std::cout<<"\"" #T "\":"<<sizeof(T)<<','
 SIZE(live602::Config);SIZE(live602::State);SIZE(live602::Action);SIZE(live602::Result);SIZE(live602::Live602Runtime);
 SIZE(Prefetcher);SIZE(Stream_Tracker);SIZE(Streamer);SIZE(Bitmap);SIZE(AMPM_PB_Entry);SIZE(AMPM);
 SIZE(SIGNATURE_TABLE);SIZE(PATTERN_TABLE);SIZE(PREFETCH_FILTER);SIZE(GLOBAL_REGISTER);
 std::cout<<"\"SPP_dev2\":"<<sizeof(SPP_dev2)<<"},\"model\":{";
 std::cout<<"\"parameters\":"<<config.parameter_count<<",\"fp32_weight_bytes\":"<<weights<<",\"global_hc_bytes\":"<<state_bytes<<",\"hidden_size\":"<<config.hidden_size<<",\"features\":"<<config.feature_count<<",\"fill_classes\":"<<config.fill_classes<<"},";
 std::cout<<"\"host_allocations\":{\"runtime_object_bytes\":"<<sizeof(live602::Live602Runtime)<<",\"persistent_requested_bytes\":"<<persistent<<",\"object_plus_persistent_requested_bytes\":"<<persistent+sizeof(live602::Live602Runtime)<<",\"constructor_peak_requested_bytes\":"<<constructor_peak<<",\"constructor_allocation_calls\":"<<constructor_calls<<",\"max_infer_incremental_peak_requested_bytes\":"<<max_incremental_peak<<",\"max_returned_result_requested_bytes\":"<<max_returned_live<<",\"max_actions_vector_capacity\":"<<max_actions_capacity<<",\"inference_allocation_calls\":"<<inference_allocation_calls<<",\"residual_after_runtime_destruction\":"<<residual<<"},";
 std::cout<<"\"fixture\":{\"callbacks\":"<<callbacks<<",\"demand_callbacks\":"<<demands<<",\"fill_callbacks\":"<<fills<<",\"actions\":"<<actions<<",\"max_count\":"<<max_count<<",\"macs\":"<<macs<<",\"nonlinears\":"<<nonlinears<<",\"first_callback\":{\"incremental_peak_requested_bytes\":"<<first_peak<<",\"returned_result_requested_bytes\":"<<first_retained<<",\"count\":"<<first_count<<",\"macs\":"<<first_macs<<",\"nonlinears\":"<<first_nonlinears<<"}},";
 std::cout<<"\"scope\":\"all ordinary C++ new/new[] requested payload bytes in scalar runtime construction and Infer; fixture/measurement/log allocations excluded; malloc metadata/alignment tracking headers/C-library allocations/RSS excluded; observed fixture peak is not a universal degree bound or hardware PPA\"}\n";
 return 0;
 }catch(const std::exception& error){allocation::enabled=false;std::cerr<<error.what()<<'\n';return 1;}}
