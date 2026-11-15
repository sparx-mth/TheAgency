import base64,collections,glob,json,os,random,sys,time
import cv2,requests
sys.path.insert(0,os.path.expanduser('~/GIT/TheAgency'))
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep import evaluate as E
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import DETECTOR_KIND
URL='http://127.0.0.1:11435/api/chat'; MODEL='qwen3-vl:4b-instruct'; PER_CELL=10; SEED=31
QUERY={'bed':'bed','sofa':'couch','chair':'chair','tv':'tv','toilet':'toilet','plant':'potted plant'}
SW=os.path.expanduser('~/detector_sweep_train'); OUT=os.path.expanduser('~/tc_sweep/vlm_boxes'); os.makedirs(OUT+'/crops',exist_ok=True)
pool=collections.defaultdict(list)
for d in sorted(glob.glob(SW+'/*/gt.json')):
    folder=os.path.dirname(d); scene=os.path.basename(folder); det=json.load(open(folder+'/det_yolo_world.json'))
    for r in json.load(open(d)):
        boxes=[b for b in r['boxes'] if b['pixels']>=E.MIN_GT_PIXELS]
        for x in E.merged_by_kind(det.get(r['frame'],[])):
            kind=DETECTOR_KIND[x['cls']]
            if x['conf']<0.30 or kind not in QUERY: continue
            tp=any(b['kind']==kind and E.iou(x['xyxy'],b['xyxy'])>=E.MATCH_IOU for b in boxes)
            pool[(kind,tp)].append((scene,r['frame'],x['xyxy'],x['conf'],kind))
rng=random.Random(SEED); items=[]
for key in sorted(pool):
    rng.shuffle(pool[key]); items+= [(key[1],)+t for t in pool[key][:PER_CELL]]
rng.shuffle(items)
def crop_with_bbox(img,xyxy,margin=0.5,min_side=224):
    h,w=img.shape[:2]; x0,y0,x1,y1=xyxy; cx,cy=(x0+x1)/2,(y0+y1)/2
    hw=max((x1-x0)*(0.5+margin),min_side/2); hh=max((y1-y0)*(0.5+margin),min_side/2)
    l,r,t,b=int(max(0,cx-hw)),int(min(w,cx+hw)),int(max(0,cy-hh)),int(min(h,cy+hh))
    c=img[t:b,l:r].copy(); cv2.rectangle(c,(int(x0-l),int(y0-t)),(int(x1-l),int(y1-t)),(0,0,255),2); return c
done={tuple(json.loads(l)['id']) for l in open(OUT+'/results.jsonl')} if os.path.exists(OUT+'/results.jsonl') else set()
out=open(OUT+'/results.jsonl','a')
for i,(tp,scene,frame,xyxy,conf,kind) in enumerate(items):
    if (scene,frame,*[round(v) for v in xyxy]) in done: continue
    img=cv2.imread(f'{SW}/{scene}/frames/{frame}'); c=crop_with_bbox(img,xyxy); path=f'{OUT}/crops/{i:03d}_{kind}_{"tp" if tp else "fp"}.png'; cv2.imwrite(path,c)
    prompt='In this photo, look only at the object inside the red rectangle. Is it a %s? Answer with JSON {"answer": "yes"} or {"answer": "no"}.'%QUERY[kind]
    body={'model':MODEL,'stream':False,'think':False,'format':'json','options':{'temperature':0,'num_predict':30,'num_ctx':4096},'messages':[{'role':'user','content':prompt,'images':[base64.b64encode(cv2.imencode('.png',c)[1].tobytes()).decode()]}]}
    t0=time.monotonic()
    try: ans=json.loads(requests.post(URL,json=body,timeout=600).json()['message']['content']).get('answer')
    except Exception as ex: ans='ERROR '+repr(ex)[:50]
    out.write(json.dumps({'id':[scene,frame,*[round(v) for v in xyxy]],'i':i,'kind':kind,'tp':tp,'conf':conf,'answer':ans,'crop':path,'s':round(time.monotonic()-t0,1)})+'\n'); out.flush()
print('VLM_BOX_DONE',len(items))
