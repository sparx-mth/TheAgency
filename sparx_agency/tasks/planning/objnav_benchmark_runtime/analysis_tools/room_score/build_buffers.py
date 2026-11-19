import collections,glob,gzip,json,math,os,re
import numpy as np
RADIUS=1.2; HOME=os.path.expanduser('~')
CONTENT=HOME+'/datasets_v01/objectnav/hm3d/v1/train/content/'
CATS=('chair','bed','plant','toilet','tv_monitor','sofa')
def box(o): return (o['min'][1],o['max'][1],-o['max'][0],-o['min'][0])
def dist(x,y,b): return math.hypot(max(b[0]-x,0,x-b[1]),max(b[2]-y,0,y-b[3]))
out=[]; stats=collections.Counter()
for run in ('trace_train_a','trace_holdout'):
    for d in sorted(glob.glob(f'{HOME}/{run}/*/')):
        scene=os.path.basename(d.rstrip('/')); stem=scene.split('-')[1]
        gtf=f'{HOME}/tc_sweep/gt/gtbox_{scene}.json'
        if not os.path.exists(gtf): continue
        gt=json.load(open(gtf)); byid={o['id']:o for o in gt}
        zf=collections.defaultdict(list)
        for o in gt:
            if o['name']=='floor': zf[o['region']].append((o['min'][2]+o['max'][2])/2)
        floors=[(box(o),o['region'],float(np.median(zf[o['region']]))) for o in gt if o['name']=='floor']
        def region(x,y,z):
            best=None
            for b,r,fz in floors:
                if abs(fz-z)>0.8: continue
                dd=dist(x,y,b)
                if dd<=RADIUS and (best is None or dd<best[0]): best=(dd,r)
            return best[1] if best else None
        goals=json.load(gzip.open(CONTENT+stem+'.json.gz'))['goals_by_category']
        goal_regions={c:{byid[g['object_id']]['region'] for g in goals.get(stem+'.basis.glb_'+c,[]) if g['object_id'] in byid} for c in CATS}
        snaps={}; det=collections.defaultdict(list); zs=collections.defaultdict(list); target_of={}
        for l in open(d+'trace.jsonl'):
            r=json.loads(l)
            if r.get('snapshot'): snaps[r['episode']]=r['snapshot']
            if r['kind']=='detect':
                m=re.search(r"category='([a-z_]+)', query='([^']+)'",r.get('target','') or '')
                if m and r['episode'] not in target_of: target_of[r['episode']]={'category':m.group(1),'query':m.group(2)}
                if r.get('pose'): zs[r['episode']].append(r['pose']['z'])
                if r.get('frame'): det[r['episode']].append(r)
        for ep,s in snaps.items():
            z=float(np.median(zs[ep])) if zs[ep] else 0.0
            for pid,v in s['rooms'].items():
                if len(v['cells'])<20: continue
                c=collections.Counter(region(x,y,z) for x,y in v['cells']); reg,n=c.most_common(1)[0]
                if reg is None or n/len(v['cells'])<0.5: stats['room_no_clear_region']+=1; continue
                frames=[]
                for r in det[ep]:
                    share=(r.get('frame_rooms') or {}).get('all',{}).get(pid,0.0)
                    if share>=0.3: frames.append({'file':f'{run}/{scene}/frames/{r["frame"]}','call':r['call'],'share_in_room':round(share,3),'floor_share':round((r.get('frame_rooms') or {}).get('floor',{}).get(pid,0.0),3),
                                                  'pose':[r['pose']['x'],r['pose']['y'],r['pose']['yaw']],'n_boxes':sum(1 for b in r['boxes'] if b['conf']>=0.3),'classes':sorted({b['cls'] for b in r['boxes'] if b['conf']>=0.3})})
                if not frames: stats['room_no_frames']+=1; continue
                out.append({'run':run,'scene':scene,'episode':ep,'room':int(pid),'target':target_of.get(ep),'region':reg,'present':{c:(reg in goal_regions[c]) for c in CATS},'frames':frames})
json.dump(out,open(HOME+'/tc_sweep/buffers.json','w'))
n=len(out); fs=[len(b['frames']) for b in out]
print('buffers (room visits with frames):',n,'| episodes',len({b['episode'] for b in out}),'| scenes',len({b['scene'] for b in out}))
print('frames per buffer: median %d, mean %.1f, min %d, max %d | buffers with >=6 frames: %d, >=3: %d'%(sorted(fs)[n//2],sum(fs)/n,min(fs),max(fs),sum(f>=6 for f in fs),sum(f>=3 for f in fs)))
print('target categories present in a room (any of six):',{c:sum(b['present'][c] for b in out) for c in CATS})
print('buffers by run:',dict(collections.Counter(b['run'] for b in out)),'| skipped:',dict(stats))
print('total frames (not unique):',sum(fs),'| unique frame files:',len({f['file'] for b in out for f in b['frames']}))
