"""Frozen true AMPM v7 reference: one global chronological h/c, no PC input."""
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from formal_NN_training.common.threshold_free_policy import build_model,_coordinate_to_delta,apply_signed_line_delta

class AMPMReference:
    def __init__(self,checkpoint):
        torch.set_num_threads(1)
        value=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (value.get('matched_normal_prefetcher'),value.get('hidden_size'),value.get('feature_count'),value.get('parameters'))!=('ampm',8,64,2821):
            raise ValueError('not the retained real AMPM v7 h8 model')
        self.model,_=build_model('lstm',64,8,fill_classes=0,mixture_components=4)
        self.model.load_state_dict(value['state_dict'],strict=True);self.model.eval()
        for parameter in self.model.parameters():parameter.requires_grad_(False)
        self.Reset()
    def Reset(self):self.state=None
    def Infer(self,address,demand=True):
        if not demand:raise ValueError('AMPM student observes eligible LOADs only')
        address=int(address)
        if address<0 or address>=2**64 or address&63:raise ValueError('aligned uint64 byte address required')
        bits=torch.tensor([[(address>>i)&1 for i in range(64)]],dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            context,self.state=self.model.encode(bits,self.state)
            context=context[0,0].unsqueeze(0)
            raw=float(self.model.decoder.count_raw(context).item())
            rate=float(np.logaddexp(0.0,np.float64(raw)))
            if not math.isfinite(rate) or rate>=2**63:raise ValueError('Poisson count outside int64 domain')
            count=int(math.floor(rate));actions=[]
            if count:
                decoder=self.model.decoder.begin(context)
                for _ in range(count):
                    mix,mean,scale,fill=self.model.decoder.distribution(decoder)
                    component=int(mix.argmax(-1).item());coordinate=mean[:,component]
                    value=float(coordinate.item());delta=_coordinate_to_delta(value)
                    actions.append(dict(address=apply_signed_line_delta(address>>6,delta)<<6,delta=delta,fill=2,fill_index=-1,coordinate=value))
                    decoder=self.model.decoder.advance(decoder,coordinate)
            return dict(address=address,demand=True,emit=count!=0,count=count,K=count,count_raw=raw,
                        hidden=self.state[0][0,0].tolist(),cell=self.state[1][0,0].tolist(),actions=actions,truncated=False)
