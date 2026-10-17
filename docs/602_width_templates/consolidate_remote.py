#!/usr/bin/env python3
"""Preserve dated Sacramento studies by same-filesystem rename, with compatibility links."""
import json, os, pathlib, time
ROOT=pathlib.Path('/home/qianruw/cache')
BASE=ROOT/'formal_study'
BASE.mkdir(exist_ok=True)
for child in ['archive','work','runs','migration']:(BASE/child).mkdir(exist_ok=True)
proof=BASE/'migration'/('remote_'+str(time.time_ns()));proof.mkdir()
maps=[];before=[]
for day in ['20261016','20261017']:
 old=ROOT/('formal_study_'+day); new=BASE/'archive'/day
 if old.is_symlink():
  assert old.resolve()==new.resolve(),str(old)
  continue
 assert old.is_dir() and not new.exists(), (old,new)
 for p in old.rglob('*'):
  if p.is_file() and not p.is_symlink():
   s=p.stat(); before.append(dict(old=str(p),new=str(new/p.relative_to(old)),size=s.st_size,dev=s.st_dev,ino=s.st_ino))
 maps.append(dict(old=str(old),new=str(new)))
(proof/'files_before.json').write_text(json.dumps(before,indent=2)+'\n')
for m in maps:
 old,new=map(pathlib.Path,[m['old'],m['new']]);old.rename(new);old.symlink_to(new.relative_to(ROOT))
for x in before:
 s=pathlib.Path(x['new']).stat();assert (s.st_size,s.st_dev,s.st_ino)==(x['size'],x['dev'],x['ino']),x
result=dict(status='PASS',regular_files=len(before),total_bytes=sum(x['size'] for x in before),same_inode_device_size=True,historical_metadata_unchanged=True,old_to_new=maps,notes=['Historical source-only directories remain archived; new work checkout is separate.','No data deleted, historical files rewritten, or authentication changed.'],proof=str(proof))
(proof/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
(BASE/'migration'/'remote_latest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
