import collections,json,math,os,random,sys
import numpy as np
from scipy.stats import rankdata
HOME=os.path.expanduser('~'); K_LIST=(1,3,5)
KEYS=['chair','bed','potted plant','toilet','tv','couch']; CAT_OF={'chair':'chair','bed':'bed','potted plant':'plant','toilet':'toilet','tv':'tv_monitor','couch':'sofa'}
DET={'chair':{'chair'},'bed':{'bed'},'potted plant':{'potted plant'},'toilet':{'toilet'},'tv':{'tv','television','monitor'},'couch':{'sofa','couch'}}
buffers=json.load(open(HOME+'/tc_sweep/buffers.json'))
S={}
for l in open(HOME+'/tc_sweep/vlm_frame_logprob.jsonl'):
    r=json.loads(l)
    if r.get('p_yes'): S[r['file']]=r['p_yes']
MINF=int(os.environ.get('MIN_FRAMES','1')); bufs=[b for b in buffers if len(b['frames'])>=MINF and all(f['file'] in S for f in b['frames'])]
print('buffers fully scored: %d of %d | frames scored: %d'%(len(bufs),len(buffers),len(S)))
def auroc(score,label):
    label=np.asarray(label,bool); s=np.asarray(score,float); n1=label.sum(); n0=(~label).sum()
    if n1<4 or n0<4: return None
    r=rankdata(s); return (r[label].sum()-n1*(n1+1)/2)/(n1*n0)
def pick(rule,fr,K,rng):
    n=len(fr); K=min(K,n)
    if rule=='all': return list(range(n))
    if rule=='random': return rng.sample(range(n),K)
    if rule=='even': return sorted({int(round(i*(n-1)/max(K-1,1))) for i in range(K)}) if K>1 else [n//2]
    key={'floor':lambda i:fr[i]['floor_share'],'room_share':lambda i:fr[i]['share_in_room'],'objects':lambda i:fr[i]['n_boxes'],
         'vlm_sum':lambda i:sum(S[fr[i]['file']].values())}.get(rule)
    if key: return sorted(range(n),key=key,reverse=True)[:K]
    if rule=='diverse':
        def d(i,j):
            a,b=fr[i]['pose'],fr[j]['pose']; return math.hypot(a[0]-b[0],a[1]-b[1])+1.5*abs(math.atan2(math.sin(a[2]-b[2]),math.cos(a[2]-b[2])))
        chosen=[max(range(n),key=lambda i:fr[i]['floor_share'])]
        while len(chosen)<K: chosen.append(max((i for i in range(n) if i not in chosen),key=lambda i:min(d(i,j) for j in chosen)))
        return chosen
AGG={'mean':lambda v:float(np.mean(v)),'max':lambda v:float(np.max(v)),'noisyor':lambda v:1-float(np.prod([1-x for x in v]))}
RULES=['random','even','floor','room_share','objects','diverse','vlm_sum']
def evaluate(blist,K,rule,agg,out_of_view):
    rng=random.Random(5); aurocs=[]
    for k in KEYS:
        sc,lb=[],[]
        for b in blist:
            fr=b['frames']
            if out_of_view and any(set(f['classes'])&DET[k] for f in fr): continue
            reps=20 if rule=='random' else 1; vals=[]
            for _ in range(reps):
                idx=pick(rule,fr,K,rng); vals.append(AGG[agg]([S[fr[i]['file']][k] for i in idx]))
            sc.append(float(np.mean(vals))); lb.append(b['present'][CAT_OF[k]])
        a=auroc(sc,lb)
        if a is not None: aurocs.append(a)
    return (float(np.mean(aurocs)),len(aurocs)) if aurocs else (None,0)
for oov in (True,False):
    print('\n=== mean AUROC over targets (%s); K frames of the buffer; agg=mean | (targets with enough positives)'%('target not seen by the detector in any buffer frame' if oov else 'all buffers'))
    print('%-11s'%'rule',*['K=%d       '%k for k in K_LIST])
    for rule in RULES+['all']:
        cells=[]
        for K in K_LIST:
            if rule=='all' and K!=K_LIST[0]: cells.append('   -      '); continue
            m,n=evaluate(bufs,K,rule,'mean',oov); cells.append('%.3f (%d)'%(m,n) if m is not None else '   n/a   ')
        print('%-11s'%rule,*cells)
print('\n=== aggregation (rule=floor, K=3, out of view):',{a:('%.3f'%evaluate(bufs,3,'floor',a,True)[0]) if evaluate(bufs,3,'floor',a,True)[0] else 'n/a' for a in AGG})
