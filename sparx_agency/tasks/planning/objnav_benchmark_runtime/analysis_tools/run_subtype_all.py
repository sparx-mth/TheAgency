import base64,glob,json,os,time
import cv2,requests
root=os.path.expanduser('~/tc_sweep'); URL='http://127.0.0.1:11435/api/chat'; MODEL='qwen3-vl:4b-instruct'
WITH_OTHER=['dining_table','desk','coffee_table','bedside_table','other_table','not_a_table']

CAB_NO_OTHER=['kitchen_cabinet','wardrobe','bathroom_cabinet','chest_of_drawers','tv_stand','shelf_or_bookcase','not_a_cabinet']
rows={(r['scene'],r['id']):r for r in map(json.loads,open(root+'/vlm_tc.jsonl'))}
cached={}
for f,k in (('table_variants.jsonl','frame_other'),('cabinet_variants.jsonl','frame_noother')):
    for r in map(json.loads,open(root+'/'+f)): cached[(r['scene'],r['id'])]=r[k]
best={}
for f in glob.glob(root+'/*/gt.json'):
    scene=f.split('/')[-2]
    for fr in json.load(open(f)):
        for b in fr['boxes']:
            x0,y0,x1,y1=b['xyxy']; touching=x0<=2 or y0<=2 or x1>=637 or y1>=477; key=(scene,b['id'])
            if key not in best or (not touching,b['pixels'])>best[key][0]: best[key]=((not touching,b['pixels']),fr['frame'],b['xyxy'])
def b64(img): return base64.b64encode(cv2.imencode('.jpg',img)[1].tobytes()).decode()
def ask(img,labels,what):
    p='In this photo of a home, the piece of furniture inside the red rectangle is %s-like. Use the surrounding room to decide which kind it is. Answer with exactly one of: %s. Reply as JSON {"kind": "<one>"}.'%(what,', '.join(labels))
    body={'model':MODEL,'stream':False,'think':False,'format':'json','options':{'temperature':0,'num_predict':60,'num_ctx':4096},'messages':[{'role':'user','content':p,'images':[b64(img)]}]}
    try: return json.loads(requests.post(URL,json=body,timeout=300).json()['message']['content']).get('kind')
    except Exception as e: return 'ERROR '+repr(e)[:50]
out=open(root+'/subtype_all.jsonl','w')
for key,r in sorted(rows.items()):
    fam=r['family']
    if key in cached: kind=cached[key]
    else:
        _,frame,xyxy=best[key]; img=cv2.imread(f'{root}/{key[0]}/frames/{frame}'); x0,y0,x1,y1=[int(v) for v in xyxy]; cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),3)
        kind=ask(img,WITH_OTHER if fam=='table' else CAB_NO_OTHER,fam)
    out.write(json.dumps({'scene':key[0],'id':key[1],'family':fam,'subtype':kind})+'\n'); out.flush()
print('SUBTYPE_DONE',len(rows))
