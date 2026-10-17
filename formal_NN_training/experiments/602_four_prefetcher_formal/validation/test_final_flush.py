#!/usr/bin/env python3
"""Use the real statistics flush prefix without destroying its stream owner."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def verify(source,out,cxx):
    source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    text=source.read_text();begin=text.index('std::string Four602::json()const {')
    begin=text.index('{',begin)+1;end=text.index(' std::ostringstream out;',begin)
    prefix=text[begin:end]
    if prefix.count('inputs_.flush()')!=1 or 'if(!inputs_)throw' not in prefix:
        raise RuntimeError('Expected one explicit checked statistics flush')
    (out/'actual_flush_prefix.cc.txt').write_text(prefix)
    shutil.copy2(source,out/'source_four602_engine.cc')
    fixture=out/'final_flush.cc'
    fixture.write_text('''#include <fstream>
#include <string>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <cstdlib>
struct Harness {
 mutable std::ofstream inputs_;
 explicit Harness(const char* path):inputs_(path){if(!inputs_)throw std::runtime_error("open");inputs_<<"sequence,value\\n";}
 ~Harness(){std::abort();} // The test must never rely on owner destruction.
 std::string json()const {
'''+prefix+'''
 return "{}";
 }
};
int main(int argc,char** argv) {
 if(argc!=3)return 2;
 Harness* writer=new Harness(argv[1]);const unsigned rows=10003;
 for(unsigned i=0;i<rows;++i)writer->inputs_<<i<<",tail_record_"<<i<<"\\n";
 writer->json();
 std::ifstream input(argv[1],std::ios::binary);
 std::string actual((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
 if(actual.empty()||actual.back()!='\\n')return 3;
 unsigned lines=0;for(char c:actual)if(c=='\\n')++lines;
 if(lines!=rows+1)return 4;
 const std::string last="10002,tail_record_10002\\n";
 if(actual.size()<last.size()||actual.substr(actual.size()-last.size())!=last)return 5;
 Harness* broken=new Harness(argv[2]);broken->inputs_.setstate(std::ios::badbit);bool rejected=false;
 try{broken->json();}catch(const std::runtime_error&){rejected=true;}
 if(!rejected)return 6;
 std::cout<<"{\\\"status\\\":\\\"PASS\\\",\\\"data_rows\\\":10003,\\\"lines_with_header\\\":10004,\\\"last_newline\\\":true,\\\"owner_not_destroyed\\\":true,\\\"failed_open_stream_rejected\\\":true}"<<std::endl;
 std::_Exit(0);
}
''')
    binary=out/'final_flush'
    command=[cxx,'-std=c++11','-O3','-ffp-contract=off',str(fixture),'-o',str(binary)]
    compiled=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/'compiler.log').write_text(compiled.stdout)
    if compiled.returncode:raise RuntimeError('Flush harness compilation failed: '+str(out/'compiler.log'))
    run=subprocess.run([str(binary),str(out/'callbacks.csv'),str(out/'failed_stream.csv')],text=True,capture_output=True)
    (out/'run.log').write_text(run.stdout+run.stderr)
    if run.returncode:raise RuntimeError('Final-stats no-destructor flush harness failed with code '+str(run.returncode))
    report=json.loads(run.stdout);report.update(actual_source=str(source),compiler_command=command,no_simulator_execution=True)
    if source.read_bytes()!=(out/'source_four602_engine.cc').read_bytes():raise RuntimeError('Flush source changed during validation')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cxx',default='g++')
    a=p.parse_args();verify(a.source,a.out,a.cxx)
