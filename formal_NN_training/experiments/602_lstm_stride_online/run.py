#!/usr/bin/env python3
"""Small entry point for the bounded Sacramento online-learning experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import tarfile
import time

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
CONFIG = json.loads((EXP / 'config.json').read_text())
RAW = Path(CONFIG['raw'])


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def build():
    builds = RAW / 'builds'
    builds.mkdir(parents=True, exist_ok=True)
    i = 1
    while (builds / ('build_%02d' % i)).exists():
        i += 1
    destination = builds / ('build_%02d' % i)
    simulator = destination / 'simulator'
    simulator.mkdir(parents=True)
    active = Path(CONFIG['simulator'])
    archive = subprocess.check_output(['git', '-C', str(active), 'archive', CONFIG['simulator_base']])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(simulator)
    # Preserve the inspected dirty cache patch; never mutate or clean active.
    subprocess.run(['patch', '-p1', '-i', str(EXP / 'patches/inherited_cache.patch')], cwd=simulator, check=True)
    shutil.copytree(active / 'libbf', simulator / 'libbf', dirs_exist_ok=True)
    (simulator / '.online602_isolated').write_text('isolated source-only build\n')
    subprocess.run([sys.executable, str(EXP / 'runtime/prepare_forward.py'), str(simulator)], check=True)
    subprocess.run([sys.executable, str(EXP / 'patches/install.py'), str(simulator)], check=True)
    with (destination / 'build.log').open('w') as log:
        subprocess.run(['bash', './build_champsim.sh', 'no', 'online602', 'no', '1'], cwd=simulator, stdout=log, stderr=subprocess.STDOUT, check=True)
    binary = simulator / 'bin/perceptron-no-online602-no-ship-1core'
    if not binary.is_file():
        raise RuntimeError('expected simulator binary missing')
    write(RAW / 'build.json', {'binary': str(binary), 'simulator': str(simulator)})
    print('Built', binary, flush=True)


def specs():
    runs = []
    def add(arm, h=0, seed=7, capacity=0, macs=0, protocol='cold'):
        rid = '{}_{}_h{}_s{}_state{}_mac{}'.format(protocol,arm,h,seed,capacity,macs)
        runs.append(dict(run_id=rid,arm=arm,h=h,hidden_size=h,seed=seed,state_capacity=capacity,macs=macs,protocol=protocol))
    add('none'); add('stride')
    for h in (8,16):
        add('frozen_i1m',h);add('frozen_i20m',h)
        for seed in (7,17,27): add('online_random',h,seed)
        add('online_warm',h)
    for macs in (0,4,16):
        add('frozen_i1m',8,capacity=64,macs=macs)
        add('online_random',8,capacity=64,macs=macs)
    for h in (8,16):
        for arm in ('frozen_i1m','frozen_i20m'):add(arm,h,protocol='historical')
    return runs


def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def run_one(spec, pilot=False):
    if (RAW/'STOP').exists():
        print('SKIP stopped',spec['run_id'],flush=True);return
    root = RAW / ('pilot' if pilot else 'final')
    path = root / spec['run_id']
    if path.exists():
        prior = json.loads((path/'run.json').read_text()) if (path/'run.json').exists() else {}
        if prior.get('status') == 'complete' and (path/'run.log').is_file() and 'ChampSim completed all CPUs' in (path/'run.log').read_text():
            print('SKIP complete',spec['run_id'],flush=True);return
        if prior.get('status')=='running' and alive(prior.get('pid',0)):
            print('SKIP running',spec['run_id'],flush=True);return
        archive = RAW/'failed'/('{}-{}'.format(spec['run_id'],time.time_ns()))
        archive.parent.mkdir(parents=True,exist_ok=True);path.rename(archive)
    path.mkdir(parents=True)
    binary = json.loads((RAW/'build.json').read_text())['binary']
    historical = spec['protocol']=='historical'
    skip = CONFIG['development_skip'] if pilot else 0 if historical else CONFIG['final_skip']
    instructions = CONFIG['pilot_instructions'] if pilot else CONFIG['final_instructions']
    warmup = 25000000 if historical else 0
    spec = dict(spec,skip_records=skip,simulation_instructions=instructions,warmup_instructions=warmup,trace=CONFIG['trace'],status='preparing')
    write(path/'run.json',spec)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    env.update(ONLINE602_METHOD='online' if spec['arm'].startswith('online') else 'frozen' if spec['arm'].startswith('frozen') else spec['arm'],
        ONLINE602_MACS=str(spec['macs']),ONLINE602_STATE_CAPACITY=str(spec['state_capacity']),
        ONLINE602_OUTPUT_LIMIT='0' if historical else str(CONFIG['decoder_limit']),
        ONLINE602_SKIP_RECORDS=str(skip),ONLINE602_MAX_RECORDS='0' if historical else str(instructions),
        ONLINE602_STATS=str(path/'snapshots.jsonl'))
    worker=None;worker_log=None;socket_path=Path('/tmp')/('online602-{}-{}.sock'.format(os.getpid(),spec['run_id']))
    if spec['h']:
        budget='i20m' if spec['arm']=='frozen_i20m' else 'i1m'
        point=Path(CONFIG['checkpoints'])/('h'+str(spec['h']))/budget/'seed7'
        env['STRIDE_LSTM_MODEL_BIN']=str(point/'export/model.bin')
        if spec['arm']!='online_random':spec['checkpoint']=str(point/'offline/model.pt')
        if spec['arm'].startswith('online'):
            command=[CONFIG['python'],str(EXP/'python/online_worker.py'),'--socket',str(socket_path),'--hidden-size',str(spec['h']),'--seed',str(spec['seed']),'--initial-model-bin',str(path/'initial_model.bin'),'--stats',str(path/'worker_stats.json')]
            if spec['arm']=='online_warm':command+=['--checkpoint',spec['checkpoint']]
            worker_log=(path/'worker.log').open('w')
            worker=subprocess.Popen(command,env=env,stdout=worker_log,stderr=subprocess.STDOUT,start_new_session=True)
            env['ONLINE602_SOCKET']=str(socket_path);env['STRIDE_LSTM_MODEL_BIN']=str(path/'initial_model.bin')
            deadline=time.monotonic()+60
            while not socket_path.exists() or not (path/'initial_model.bin').exists():
                if worker.poll() is not None or time.monotonic()>deadline:
                    if worker.poll() is None:os.killpg(worker.pid,signal.SIGTERM);worker.wait(timeout=10)
                    worker_log.close()
                    raise RuntimeError('worker startup failed: '+str(path))
                time.sleep(.05)
    command=[binary,'--l2c_prefetcher_types=online602','--stride_num_trackers=64','--stride_pref_degree=2','--warmup_instructions='+str(warmup),'--simulation_instructions='+str(instructions),'-traces',CONFIG['trace']]
    start=time.monotonic();spec['command']=command;spec['worker_pid']=worker.pid if worker else None
    print('START',spec['run_id'],flush=True)
    try:
        with (path/'run.log').open('w') as log:
            process=subprocess.Popen(['/usr/bin/time','-v','-o',str(path/'host_time.txt')]+command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            spec.update(status='running',pid=process.pid);write(path/'run.json',spec)
            code=process.wait()
        complete=code==0 and 'ChampSim completed all CPUs' in (path/'run.log').read_text()
        spec.update(status='complete' if complete else 'failed',exit_code=code,host_seconds=time.monotonic()-start)
        timing=(path/'host_time.txt').read_text()
        rss=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',timing)
        spec['simulator_wall_seconds']=spec['host_seconds']
        spec['simulator_peak_rss_kib']=int(rss.group(1)) if rss else None
        write(path/'run.json',spec)
        if not complete:raise RuntimeError('simulation failed: '+str(path))
        print('COMPLETE',spec['run_id'],'seconds',round(spec['host_seconds'],2),flush=True)
    finally:
        if worker:
            try:worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid,signal.SIGTERM);worker.wait(timeout=10)
            worker_log.close()
        if socket_path.exists():socket_path.unlink()


def run(pilot=False, select=None):
    selected=specs()
    if pilot:
        selected=[dict(run_id='development_online_h8_s7_mac0',arm='online_random',h=8,hidden_size=8,seed=7,state_capacity=0,macs=0,protocol='cold')]
    if select:selected=[s for s in selected if select in s['run_id']]
    if not selected:raise RuntimeError('no selected runs')
    if not pilot and not (RAW/'recipe_locked.json').exists():raise RuntimeError('pilot and recipe lock required before final evaluation')
    errors=[]
    with ThreadPoolExecutor(max_workers=min(CONFIG['jobs'],2)) as pool:
        futures=[pool.submit(run_one,s,pilot) for s in selected]
        for future in as_completed(futures):
            try:future.result()
            except Exception as error:errors.append(str(error));print('ERROR',error,flush=True)
    if errors:raise RuntimeError('\n'.join(errors))


def status():
    for path in sorted(RAW.glob('*/*/run.json')):
        r=json.loads(path.read_text());snap=path.parent/'snapshots.jsonl';last={}
        if snap.exists():
            for line in snap.read_text().splitlines():
                try:
                    value=json.loads(line)
                    if value.get('event')=='snapshot':last=value
                except json.JSONDecodeError:pass
        print(r.get('run_id'),r.get('status'),'instructions',last.get('instructions',0),'callbacks',last.get('eligible_l2_callbacks',0),'updates',last.get('completed_updates',0),flush=True)


def tests():
    for script in sorted((EXP/'validation').glob('test_*.py')):
        subprocess.run([CONFIG['python'],str(script)],cwd=ROOT,check=True)


def stop():
    (RAW/'STOP').write_text('Stop only this experiment; resume clears this marker.\n')
    for path in RAW.glob('*/*/run.json'):
        spec=json.loads(path.read_text())
        if spec.get('status')!='running':continue
        for key in ('pid','worker_pid'):
            pid=spec.get(key)
            if not pid:continue
            proc=Path('/proc')/str(pid)/'cmdline'
            if not proc.exists():continue
            cmd=proc.read_bytes().replace(b'\0',b' ').decode()
            allowed=str(RAW/'builds') in cmd if key=='pid' else str(EXP/'python/online_worker.py') in cmd
            if not allowed:raise RuntimeError('refusing to stop a reused/unrelated PID '+str(pid))
            os.killpg(pid,signal.SIGTERM)
        spec['status']='stopped';write(path,spec)
        print('STOPPED experiment process groups',spec['run_id'])


def report(args):
    run_dir=args.run_dir or RAW/'final'
    output=args.out_dir or EXP/'report'
    analysis_command=[sys.executable,str(EXP/'python/analyze.py'),'--run-dir',str(run_dir),'--out-dir',str(EXP/'results')]
    historical_no_pref=args.historical_no_pref
    if historical_no_pref is None and args.historical_root:
        candidate=args.historical_root/'references/logs/no_pref.log'
        if candidate.is_file():historical_no_pref=candidate
    if historical_no_pref:analysis_command+=['--historical-no-pref',str(historical_no_pref)]
    subprocess.run(analysis_command,check=True)
    command=[sys.executable,str(EXP/'python/report.py'),'--results-dir',str(EXP/'results'),'--out-dir',str(output),'--recipe',str(EXP/'config.json')]
    if args.historical_root:command+=['--historical-root',str(args.historical_root)]
    subprocess.run(command,check=True)
    name='602_stride_online_learning'
    if args.compile_via:
        ssh=['ssh']+(['-S',str(args.ssh_control)] if args.ssh_control else [])
        remote=str(RAW/'report_builds'/str(time.time_ns()))
        subprocess.run(ssh+[args.compile_via,'mkdir -p '+shlex.quote(remote)],check=True)
        subprocess.run(['rsync','-a','-e',shlex.join(ssh),str(output)+'/',args.compile_via+':'+remote+'/'],check=True)
        for _ in range(2):
            compile_command='cd '+shlex.quote(remote)+' && pdflatex -halt-on-error -interaction=nonstopmode '+name+'.tex > compile.stdout 2>&1'
            result=subprocess.run(ssh+[args.compile_via,compile_command])
            if result.returncode:
                subprocess.run(['rsync','-a','-e',shlex.join(ssh),args.compile_via+':'+remote+'/compile.stdout',str(output/'compile.stdout')],check=True)
                raise RuntimeError('TeX compilation failed; inspect '+str(output/'compile.stdout'))
        subprocess.run(['rsync','-a','-e',shlex.join(ssh),args.compile_via+':'+remote+'/'+name+'.pdf',str(output/name)+'.pdf'],check=True)
    else:
        for _ in range(2):subprocess.run(['pdflatex','-halt-on-error','-interaction=nonstopmode',name+'.tex'],cwd=output,check=True)
    print('Compiled report',output/(name+'.pdf'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','test','pilot','run','status','resume','report','lock','stop']);p.add_argument('--select')
    p.add_argument('--run-dir',type=Path);p.add_argument('--out-dir',type=Path);p.add_argument('--historical-root',type=Path)
    p.add_argument('--historical-no-pref',type=Path,help='Original same-protocol no-prefetch end log; never used for cold or dynamic comparisons')
    p.add_argument('--compile-via');p.add_argument('--ssh-control',type=Path);a=p.parse_args()
    if a.action=='build':build()
    elif a.action=='test':tests()
    elif a.action=='pilot':run(True,a.select)
    elif a.action in ('run','resume'):
        if a.action=='resume' and (RAW/'STOP').exists():(RAW/'STOP').unlink()
        run(False,a.select)
    elif a.action=='status':status()
    elif a.action=='stop':stop()
    elif a.action=='lock':
        pilot=RAW/'pilot/development_online_h8_s7_mac0/run.json'
        if not pilot.exists() or json.loads(pilot.read_text()).get('status')!='complete':raise RuntimeError('complete pilot required')
        if (RAW/'recipe_locked.json').exists():raise RuntimeError('recipe already locked')
        write(RAW/'recipe_locked.json',dict(CONFIG,locked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),runs=specs()))
    elif a.action=='report':report(a)


if __name__=='__main__':main()
