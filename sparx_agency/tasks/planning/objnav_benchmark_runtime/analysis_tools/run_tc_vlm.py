import base64,glob,json,os,time
import cv2,requests
root=os.path.expanduser('~/tc_sweep'); URL='http://127.0.0.1:11435/api/chat'; MODEL='qwen3-vl:4b-instruct'
TABLE=['dining_table','desk','coffee_table','bedside_table','other_table','not_a_table']
CAB=['kitchen_cabinet','wardrobe','bathroom_cabinet','chest_of_drawers','tv_stand','shelf_or_bookcase','other_cabinet','not_a_cabinet']
def family(name): return 'table' if ('table' in name or 'desk' in name) else 'cabinet'
best={}
for f in glob.glob(root+'/*/gt.json'):
    scene=f.split('/')[-2]
    for r in json.load(open(f)):
        for b in r['boxes']:
            x0,y0,x1,y1=b['xyxy']; touching=x0<=2 or y0<=2 or x1>=637 or y1>=477
            key=(scene,b['id']); score=(not touching,b['pixels'])
            if key not in best or score>best[key][0]: best[key]=(score,scene,r['frame'],b)
path=root+'/vlm_tc.jsonl'; done={(r['scene'],r['id']) for r in map(json.loads,open(path))}
out=open(path,'a'); os.makedirs(root+'/crops',exist_ok=True)
for (scene,iid),(score,_,frame,b) in sorted(best.items()):
    if (scene,iid) in done: continue
    img=cv2.imread(f'{root}/{scene}/frames/{frame}'); h,w=img.shape[:2]
    x0,y0,x1,y1=b['xyxy']; px,py=0.25*(x1-x0),0.25*(y1-y0)
    crop=img[int(max(0,y0-py)):int(min(h,y1+py)),int(max(0,x0-px)):int(min(w,x1+px))]
    cpath=f'{root}/crops/{scene[:5]}_{iid}.jpg'; cv2.imwrite(cpath,crop)
    fam=family(b['name'].lower()); labels=TABLE if fam=='table' else CAB
    prompt='This is a cropped photo of a piece of furniture from a home. Which kind is it? Answer with exactly one of: %s. Reply as JSON {"kind": "<one>"}.'%', '.join(labels)
    body={'model':MODEL,'stream':False,'think':False,'format':'json','options':{'temperature':0,'num_predict':60,'num_ctx':4096},
          'messages':[{'role':'user','content':prompt,'images':[base64.b64encode(open(cpath,'rb').read()).decode()]}]}
    t=time.monotonic()
    try: kind=json.loads(requests.post(URL,json=body,timeout=300).json()['message']['content']).get('kind')
    except Exception as e: kind='ERROR '+repr(e)[:60]
    out.write(json.dumps({'scene':scene,'id':iid,'gt_name':b['name'],'family':fam,'pixels':b['pixels'],'crop':cpath,'vlm':kind,'s':round(time.monotonic()-t,1)})+'\n'); out.flush()
print('TC_VLM_DONE')
