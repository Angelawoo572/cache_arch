"""Check the C++ diagnostic against every grant in the Python physical stream."""
import argparse,csv,os,subprocess,tempfile
from pathlib import Path
from extract import CACHE
HERE=Path(__file__).resolve().parent
SOURCE='''
#include "frozen_rule.h"
#include <iostream>
int main() {
    uint64_t pc, line, n=0; wide_probe::Engine e;
    while(std::cin>>pc>>line) {
        auto t=e.step(pc,line);
        if(e.granted())std::cout<<n<<" "<<t<<" "<<e.chosen_action()<<" "<<e.context_key()<<"\\n";
        ++n;
    }
    std::cerr<<"seen pages "<<e.seen_pages()<<" seen PCs "<<e.seen_pcs()<<" trained contexts "<<e.trained_contexts()<<"\\n";
}
'''

def main(root):
    with (root/'r2_reference.csv').open() as f:expected=list(csv.DictReader(f))
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        p=Path(tmp);(p/'check.cc').write_text(SOURCE)
        subprocess.run([os.environ.get('CXX','c++'),'-std=c++14','-O2','-I'+str(HERE),str(p/'check.cc'),'-o',str(p/'check')],check=True)
        for mode in ('recent','plus128'):
            with (root/'r2_ingress.txt').open() as stream:
                out=subprocess.run([str(p/'check')],env=dict(os.environ,WIDE_MODE=mode,WIDE_TABLE=str(root/'r1_table.txt')),stdin=stream,
                                   stdout=subprocess.PIPE,universal_newlines=True,check=True)
            rows=[list(map(int,x.split())) for x in out.stdout.splitlines()]
            if len(rows)!=len(expected):raise AssertionError('grant count mismatch')
            for (i,t,a,k),ref in zip(rows,expected):
                cols=['ingress_index','target_line','action_index','context_key'] if mode=='recent' else ['ingress_index','plus128_target']
                if [i,t,a,k][:len(cols)]!=[int(ref[c]) for c in cols]:
                    raise AssertionError('mismatch {}, {} {}'.format(mode,(i,t,a,k),ref))
            print(mode,len(rows),'grants matched')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=CACHE/'cactus_live')
    main(ap.parse_args().out)
