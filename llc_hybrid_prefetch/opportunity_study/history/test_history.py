import unittest
import numpy as np
from stream import build,contexts,B1,B2,B3,PERMS
from evaluate import guarded,ceilings


def fixture():
    n=220;ix=np.arange(n)
    addresses=np.array([6400,6401,6528,6403,6401,6529,6400,6528]*28)[:n]
    return dict(t=ix*80,pa=addresses,va=ix,pc=np.where(ix%3,100,108),instr=ix+1000,
                op=ix*0,state=ix%4,tag_t=ix*80+150,hit=ix%2,baseline=ix*0,head=ix%5,
                proposed=ix*0-1,dispatch=ix*80+40,fill=ix*80+200,pf_gen=ix*0-1,
                pf_fill=ix*0-1,pf_kind=ix*0)


class HistoryTests(unittest.TestCase):
    def test_future_and_virtual_perturbations(self):
        z=fixture();b=build(z);alt={k:v.copy() for k,v in z.items()}
        alt['va']+=np.arange(len(alt['t']))*31337
        cut=87
        alt['pa'][cut:]+=9999;alt['pc'][cut:]+=1000;alt['state'][cut:]+=10
        alt['hit'][alt['tag_t']>=alt['t'][cut]]^=1
        bb=build(alt)
        np.testing.assert_array_equal(b['cand'][:cut],bb['cand'][:cut])
        for (name,a),(other,c) in zip(contexts(z,b),contexts(alt,bb)):
            self.assertEqual(name,other);np.testing.assert_array_equal(a[:cut],c[:cut])

    def test_label_is_first_touch_and_strict(self):
        z=fixture();b=build(z)
        for j in range(25):
            for p,k in zip(b['cand'][j],b['nxt'][j]):
                if k<0:continue
                self.assertEqual(z['pa'][k],p);self.assertFalse(np.any(z['pa'][j+1:k]==p))
                self.assertGreater(k,j);self.assertLessEqual(k-j,64)
        self.assertEqual(b['last_hit'][1],0) # first lookup has not completed
        self.assertEqual(b['last_hit'][2],1+z['hit'][0])
        prior=set()
        for pos,cs in enumerate(b['cand']):
            current=z['pa'][pos];prior.add(current//64)
            self.assertTrue(all(x//64 in prior for x in cs if x>=0))
            self.assertTrue(all(x//64==current//64 for x in cs[:B1] if x>=0))
            self.assertTrue(all(x//64!=current//64 for x in cs[B1:B3] if x>=0))

    def test_no_heldout_label_crosses_boundary(self):
        z=fixture();g=guarded(z,0,132)
        self.assertGreater(len(g),0)
        self.assertTrue(np.all(g+64<132));self.assertTrue(np.all(z['t'][g]+4096<=z['t'][131]))
        for w,rows in PERMS.items():
            for row in rows:self.assertEqual(sorted(row),list(range(w)))


if __name__=='__main__':
    unittest.main()
