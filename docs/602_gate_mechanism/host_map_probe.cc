#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <malloc.h>
#include <string>
#include <utility>
static uint64_t bytes_requested=0,bytes_usable=0,allocations=0;
template<class T>struct CountingAllocator {
  using value_type=T;
  CountingAllocator() noexcept {}
  template<class U>CountingAllocator(const CountingAllocator<U>&) noexcept {}
  template<class U>struct rebind {using other=CountingAllocator<U>;};
  T* allocate(std::size_t n){T* p=static_cast<T*>(::operator new(n*sizeof(T)));bytes_requested+=n*sizeof(T);bytes_usable+=malloc_usable_size(p);++allocations;return p;}
  void deallocate(T* p,std::size_t) noexcept {::operator delete(p);}
};
template<class T,class U>bool operator==(const CountingAllocator<T>&,const CountingAllocator<U>&){return true;}
template<class T,class U>bool operator!=(const CountingAllocator<T>&,const CountingAllocator<U>&){return false;}
int main(int argc,char** argv){
  using Pair=std::pair<const uint64_t,bool>;
  using Map=std::map<uint64_t,bool,std::less<uint64_t>,CountingAllocator<Pair>>;
  const uint64_t entries=argc>1?std::stoull(argv[1]):64;
  Map table;for(uint64_t i=0;i<entries;++i)table.emplace(i,(i&1)!=0);
  std::cout<<"{\"compiler\":\""<<__VERSION__<<"\",\"glibcxx\":"<<__GLIBCXX__<<",\"pointer_bytes\":"<<sizeof(void*)
    <<",\"std_map_object_bytes\":"<<sizeof(std::map<uint64_t,bool>)<<",\"counting_map_object_bytes\":"<<sizeof(Map)
    <<",\"std_pair_pc_bool_bytes\":"<<sizeof(Pair)<<",\"entries\":"<<table.size()<<",\"node_allocations\":"<<allocations
    <<",\"node_requested_bytes_total\":"<<bytes_requested<<",\"node_malloc_usable_bytes_total\":"<<bytes_usable
    <<",\"map_object_plus_requested_nodes_bytes\":"<<sizeof(Map)+bytes_requested
    <<",\"map_object_plus_usable_nodes_bytes\":"<<sizeof(Map)+bytes_usable
    <<",\"std_string_object_bytes\":"<<sizeof(std::string)<<"}\n";
}
