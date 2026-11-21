import json,math,os,random,sys
import numpy as np
from scipy.stats import rankdata
H=os.path.expanduser('~/tc_sweep/'); B=json.load(open(H+'buffers.json'))
KEYS=['chair','bed','potted plant','toilet','tv','couch']; CAT={'chair':'chair','bed':'bed','potted plant':'plant','toilet':'toilet','tv':'tv_monitor','couch':'sofa'}
DET={'chair':{'chair'},'bed':{'bed'},'potted plant':{'potted plant'},'toilet':{'toilet'},'tv':{'tv','television','monitor'},'couch':{'sofa','couch'}}
def load(p):
    d={}
    for l in open(p):
        r=json.loads(l)
        if r.get('p_yes') is not None: d[r['file']]=r['p_yes']
    return d
SIX={'old':load(H+'vlm_frame_logprob.jsonl'),'A':load(H+'vlm_frame_six_2b_A.jsonl'),'B':load(H+'vlm_frame_six_2b_B.jsonl')}
ONE={'2B old prompt':SIX['old'],'2B A prior':load(H+'vlm_frame_target_A.jsonl'),'2B B search':load(H+'vlm_frame_target_B.jsonl'),
     '4B A prior':load(H+'vlm_frame_target4b_A.jsonl'),'4B B search':load(H+'vlm_frame_target4b_B.jsonl')}
def auc(s,l,m=3):
    s=np.asarray(s,float); l=np.asarray(l,bool); k=~np.isnan(s); s,l=s[k],l[k]; n1=l.sum(); n0=(~l).sum()
    if n1<m or n0<m: return None
    r=rankdata(s); return (r[l].sum()-n1*(n1+1)/2)/(n1*n0)
def mean_over_frames(b,get):
    v=[x for f in b['frames'] for x in [get(f['file'])] if x is not None and not math.isnan(x)]; return float(np.mean(v)) if v else float('nan')
def episode_target_table(subset,name):
    print(name,'(rooms %d, holding the target %d)'%(len(subset),sum(b['present'][b['target']['category']] for b in subset)))
    lab=[b['present'][b['target']['category']] for b in subset]; scenes=sorted({b['scene'] for b in subset}); rng=random.Random(3); sc={}
    for n,src in ONE.items():
        sc[n]=[mean_over_frames(b,(lambda f,q=b['target']['query'],s=src,six=(src is SIX['old']):(s[f][q] if six else s[f]) if f in s else None)) for b in subset]
    for n in ONE:
        a=auc(sc[n],lab); bs=[]
        for _ in range(400):
            idx=[i for p in [rng.choice(scenes) for _ in scenes] for i,b in enumerate(subset) if b['scene']==p]
            z=auc([sc[n][i] for i in idx],[lab[i] for i in idx]) if len(idx)>5 else None
            if z is not None: bs.append(z)
        print('  %-14s AUROC %.3f  [%.3f, %.3f]'%(n,a,*np.percentile(bs,[2.5,97.5])))
def six_table(subsets,name):
    print(name); print('  %-13s old        A          B'%'target')
    means={v:[] for v in SIX}
    for k in KEYS:
        bl=subsets[k]; lab=[b['present'][CAT[k]] for b in bl]; cells=[]
        for v,src in SIX.items():
            a=auc([mean_over_frames(b,lambda f,s=src:s[f][k] if f in s else None) for b in bl],lab,4); cells.append('%.3f     '%a if a else ' n/a      ')
            if a: means[v].append(a)
        print('  %-13s %s (%d/%d)'%(k,''.join(cells),sum(lab),len(lab)))
    print('  %-13s %s'%('MEAN',''.join('%.3f     '%np.mean(means[v]) for v in SIX)))
oov_one=[b for b in B if not any(set(f['classes'])&DET[b['target']['query']] for f in b['frames'])]
episode_target_table(B,'EPISODE TARGET ONLY, all room visits'); episode_target_table(oov_one,'EPISODE TARGET ONLY, rooms where the detector never saw the target')
six_table({k:B for k in KEYS},'\nSIX OBJECTS, all room visits')
six_table({k:[b for b in B if not any(set(f['classes'])&DET[k] for f in b['frames'])] for k in KEYS},'\nSIX OBJECTS, rooms where the detector never saw that object')
