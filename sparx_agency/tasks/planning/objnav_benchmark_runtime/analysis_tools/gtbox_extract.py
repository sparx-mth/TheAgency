import os,sys,json,csv,time,gc
import numpy as np,trimesh
root=os.path.expanduser('~/datasets_v01/scene_datasets/hm3d/train'); out=sys.argv[1]
for scene in (s[0] for s in json.load(open(sys.argv[2]))):
    sid=scene.split('-')[1]; t=time.time()
    rows=list(csv.reader(open(f'{root}/{scene}/{sid}.semantic.txt'),skipinitialspace=True))[1:]
    byhex={int(r[1],16):(int(r[0]),r[2].strip('"'),int(r[3])) for r in rows}
    sc=trimesh.load(f'{root}/{scene}/{sid}.semantic.glb',force='scene',process=False)
    acc={}; cache={}
    for node in sc.graph.nodes_geometry:
        T,gname=sc.graph[node]; g=sc.geometry[gname]; vc=g.visual
        V=trimesh.transformations.transform_points(np.asarray(g.vertices),T)
        mat=vc.material; img=getattr(mat,'baseColorTexture',None) or getattr(mat,'image',None)
        a=cache.get(id(img))
        if a is None: a=np.asarray(img.convert('RGB')); cache[id(img)]=a
        H,W=a.shape[:2]; uv=np.asarray(vc.uv); best=None
        for name,vv in (('flip',1-uv[:,1]),('noflip',uv[:,1])):
            px=np.clip((uv[:,0]*(W-1)).round().astype(int),0,W-1); py=np.clip((vv*(H-1)).round().astype(int),0,H-1)
            col=a[py,px].astype(np.int64); key=(col[:,0]<<16)|(col[:,1]<<8)|col[:,2]
            hit=np.isin(key,list(byhex.keys())).mean()
            if best is None or hit>best[0]: best=(hit,name,key)
        key=best[2]
        for k in np.unique(key):
            if k not in byhex: continue
            v=V[key==k]; lo,hi=v.min(0),v.max(0)
            acc[k]=(np.minimum(acc[k][0],lo),np.maximum(acc[k][1],hi)) if k in acc else (lo,hi)
    del sc; gc.collect()
    res=[{'id':byhex[k][0],'name':byhex[k][1],'region':byhex[k][2],'min':lo.tolist(),'max':hi.tolist()} for k,(lo,hi) in acc.items()]
    json.dump(res,open(os.path.join(out,'gtbox_%s.json'%scene),'w')); print(scene,len(res),round(time.time()-t),'s',flush=True)
