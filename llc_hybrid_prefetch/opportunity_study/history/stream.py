"""Hardware-visible candidate vocabulary and lossy content summaries."""
from collections import defaultdict, deque
import numpy as np

SAME = (-32,-16,-8,-4,-3,-2,-1,1,2,3,4,8,16,32)
WIDE = (-1024,-512,-256,-128,-96,-64,-32,-16,-8,-4,-2,-1,1,2,4,8,16,32,64,96,128,256,512,1024)
PAGE_TOKENS = ('same_offset','last_offset','last_minus_1','last_plus_1')
LABELS = (['same:%+d'%d for d in SAME]+['wide:%+d'%d for d in WIDE]+
          ['page%d:%s'%(j,s) for j in range(1,5) for s in PAGE_TOKENS]+
          ['address_successor','pc_successor','two_pc_successor','local_stride','cross_pc_delta'])
B1=len(SAME); B2=B1+len(WIDE); B3=B2+4*len(PAGE_TOKENS)
MAX_EVENTS=64
MAX_CYCLES=4096
WINDOWS=(4,8,16,32,64)
# 64 independent permutations, deterministic per anchor with no label access.
RNG=np.random.default_rng(447)
PERMS={w:np.array([RNG.permutation(w) for _ in range(64)]) for w in WINDOWS}
SALT=np.uint64(11400714819323198485)


def mix(a,b):
    with np.errstate(over='ignore'):
        return (a.astype('uint64') ^ b.astype('uint64')) * SALT


def delta_class(d):
    return np.where(d==0,0,np.where(abs(d)==1,1,np.where(abs(d)<=4,2,np.where(abs(d)<64,3,4)))) + 5*(d<0)


def lookup(table,key):
    item=table[key%len(table)]
    return item[1] if item is not None and item[0]==key else -1


def put(table,key,value):
    table[key%len(table)] = (key,value)


def build(z):
    p=z['pa']; pc=z['pc']; t=z['t']; n=len(t)
    # A core in this simulator admits at most one demand each cycle. Decline
    # unsupported logs instead of making simultaneous features available early.
    if np.any(t[1:]<=t[:-1]):
        raise ValueError('non-increasing ingress cycles within a core')
    cand=np.full((n,len(LABELS)),-1,dtype=np.int64)
    candidates_page=np.full((n,4),-1,dtype=np.int64)
    recent=[]; prevpageoff={}; first_page={}
    pcp={}; prev_pc_index=np.full(n,-1,'int32')
    taddr,tpc,tchain,tdelta=([None]*256 for _ in range(4))
    prev_line=prev_pc=prev_chain=None
    for i in range(n):
        line=int(p[i]); ip=int(pc[i]); page,off=divmod(line,64)
        if i:
            # The successor of ingress i-2 became visible at i-1. Its row
            # may be consulted now; the unknown successor of i is never used.
            if i>=2:
                older_line=int(p[i-2]); older_pc=int(pc[i-2]); olderprev=int(pc[i-3]) if i>=3 else 0
                succ=int(p[i-1]); chain=(olderprev*1000003)^older_pc
                put(taddr,older_line,succ); put(tpc,older_pc,succ); put(tchain,chain,succ)
                put(tdelta,chain,succ-older_line)
            prior_pg=int(p[i-1])//64
            prevpageoff[prior_pg]=int(p[i-1])%64
            if prior_pg in recent: recent.remove(prior_pg)
            recent.insert(0,prior_pg)
            del recent[5:]
        first_page.setdefault(page,i)
        # Wide adders only address pages already seen in this physical context.
        for j,d in enumerate(WIDE):
            target=line+d; dest=target//64
            if dest!=page and dest in first_page:
                cand[i,B1+j]=target
        older=[v for v in recent if v!=page][:4]
        for j,pg in enumerate(older):
            candidates_page[i,j]=pg
            lastoff=prevpageoff[pg]
            for k,o in enumerate((off,lastoff,lastoff-1,lastoff+1)):
                if 0<=o<64: cand[i,B2+4*j+k]=64*pg+o
        chain=((int(pc[i-1]) if i else 0)*1000003)^ip
        d=lookup(tdelta,chain)
        for j,x in enumerate((lookup(taddr,line),lookup(tpc,ip),lookup(tchain,chain),
                              2*line-int(p[pcp[ip]]) if ip in pcp else -1,
                              line+d if d!=-1 or (tdelta[chain%256] or (None,))[0]==chain else -1)):
            if x>=0 and x!=line and x//64 in first_page:
                cand[i,B3+j]=x
        if ip in pcp: prev_pc_index[i]=pcp[ip]
        pcp[ip]=i
    for j,d in enumerate(SAME):
        dest=p+d; good=dest//64==p//64; cand[good,j]=dest[good]
    # Find the next touch of each generated address. This array is only read by
    # offline targets/evaluation. Candidate construction above cannot see it.
    nxt=np.full(cand.shape,-1,'int32')
    idx=np.arange(n); packed=p*(n+1)+idx
    order=np.argsort(packed); packed=packed[order]
    for j in range(cand.shape[1]):
        x=cand[:,j]; off=np.searchsorted(packed,x*(n+1)+idx+1)
        near=order[np.minimum(off,n-1)]
        good=(x>=0)&(off<n)&(p[near]==x)&(near>idx)&(near-idx<=MAX_EVENTS)&(t[near]-t<=MAX_CYCLES)
        nxt[good,j]=near[good]
    # Read the last completed lookup in strict event time, even with reordering.
    tagged=idx[z['tag_t']>=0]; ordtag=tagged[np.argsort(z['tag_t'][tagged],kind='stable')]
    k=np.searchsorted(z['tag_t'][ordtag],t,side='left')-1
    last_hit=np.where(k>=0,z['hit'][ordtag[np.maximum(k,0)]]+1,0) if len(ordtag) else np.zeros(n,int)
    return dict(cand=cand,older=candidates_page,nxt=nxt,prev=prev_pc_index,last_hit=last_hit,
                first_page=first_page)


def contexts(z,b):
    pc=z['pc']; line=z['pa']; n=len(pc); ix=np.arange(n)
    gd=line-np.r_[line[0],line[:-1]]
    prev=b['prev']; local=line-line[np.maximum(prev,0)]; local[prev<0]=0
    gtoken=delta_class(gd)+10*(pc!=np.r_[pc[0],pc[:-1]])
    ltoken=delta_class(local)
    base=mix(mix(pc,(line%64)//8),delta_class(gd))
    yield 'recent',base
    yield 'global_frequency',np.zeros(n,dtype='uint64')
    yield 'physical_offset_delta',mix((line%64)//8,delta_class(gd))
    yield 'pc_low12',mix(mix(pc%4096,(line%64)//8),delta_class(gd))
    yield 'recent_with_outcome',mix(base,b['last_hit'])
    enriched=mix(mix(base,b['last_hit']),z['state'])
    yield 'recent_with_outcome_stride',enriched
    # Run-length bins and local/global agreement model repeated transitions.
    changed=np.r_[True,local[1:]!=local[:-1]] | np.r_[True,pc[1:]!=pc[:-1]]
    start=np.maximum.accumulate(np.where(changed,ix,0))
    run=np.minimum(7,np.floor(np.log2(1+ix-start)).astype(int))
    skip=enriched.copy()
    for lag in (8,4,1):
        skip=mix(skip,np.where(ix>=lag,(pc[np.maximum(ix-lag,0)]>>2)%16,16))
    yield 'skip_pc_association',skip
    yield 'transition_flags',mix(mix(mix(enriched,run),ltoken),np.sign(local)==np.sign(gd))
    glob=np.tile(ix[:,None],(1,64))-np.arange(64,0,-1)
    loc=np.full_like(glob,-1); loc[:,-1]=prev
    for j in range(62,-1,-1):
        v=loc[:,j+1]; loc[:,j]=np.where(v<0,-1,prev[np.maximum(v,0)])
    for channel,indices,tokens in [('cross_pc',glob,gtoken),('pc_local',loc,ltoken)]:
        hist=np.where(indices<0,20,tokens[np.maximum(indices,0)]).astype('uint8')
        for w in WINDOWS:
            window=hist[:,-w:]
            for perm in (False,True):
                value=window if not perm else np.take_along_axis(window,PERMS[w][(z['instr']*131+z['op']*17)%64],axis=1)
                # Four position-sensitive chunks. Each records which behavior
                # dominates: short forward, short backward, jump or unknown.
                blocks=value.reshape(n,4,w//4)
                types=blocks%10
                cats=np.stack([((types<=2)&(blocks!=20)).sum(2),((types>=5)&(types<=7)).sum(2),
                               ((types==3)|(types==4)|(types>=8)).sum(2),(blocks==20).sum(2)],axis=2)
                winner=cats.argmax(2)+4*(((blocks>=10)&(blocks<20)).sum(2)>=(w//4+1)//2)
                k=enriched.copy()
                for q in range(4):k=mix(k,winner[:,q])
                yield '%s_w%d_%s'%(channel,w,'shuffled' if perm else 'ordered'),k
            if w==16:
                # A simple partial association: older jump + newer small stride
                # + whether any interleaved-PC ingress occurred. This statistic
                # is intentionally lossy to retain support on infrequent patterns.
                older_jump=np.any(((window[:,:8]%10)==4)|((window[:,:8]%10)==9),axis=1)
                following=np.minimum(3,np.sum((window[:,8:]%10)<=2,axis=1))
                yield channel+'_older_jump',mix(mix(mix(enriched,older_jump),following),np.any((window>=10)&(window<20),axis=1))
