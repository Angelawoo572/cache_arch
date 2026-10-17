#include <bits/stdc++.h>
#define private public
#define protected public
#include "dynamic602_engine.h"
#undef private
#undef protected
int main(){
 std::cout<<"{\"dynamic602_object_bytes\":"<<sizeof(Dynamic602)
   <<",\"event_bytes\":"<<sizeof(Dynamic602::Event)
   <<",\"entry_bytes\":"<<sizeof(Dynamic602::Entry)
   <<",\"output_bytes\":"<<sizeof(Dynamic602::Output)
   <<",\"payload_selector_string_bytes\":"<<sizeof(((Dynamic602*)0)->payload_mode_)
   <<",\"gate_table_map_object_bytes\":"<<sizeof(((Dynamic602*)0)->gate_table_)
   <<",\"phase_word_bytes\":"<<sizeof(((Dynamic602*)0)->phase_instructions_)
   <<",\"roi_word_bytes\":"<<sizeof(((Dynamic602*)0)->roi_instructions_)
   <<",\"distribution_map_object_bytes\":"<<sizeof(((Dynamic602*)0)->mechanism_distributions_)
   <<",\"classification_map_object_bytes\":"<<sizeof(((Dynamic602*)0)->mechanism_requests_)<<"}\n";
}
