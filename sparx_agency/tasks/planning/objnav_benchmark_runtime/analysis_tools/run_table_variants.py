import base64,glob,json,os,time
import cv2,requests
root=os.path.expanduser('~/tc_sweep'); URL='http://127.0.0.1:11435/api/chat'; MODEL='qwen3-vl:4b-instruct'
WITH_OTHER=['dining_table','desk','coffee_table','bedside_table','other_table','not_a_table']
NO_OTHER=['dining_table','desk','coffee_table','bedside_table','not_a_table']
SPECIFIC={'dining table':'dining_table','kitchen table':'dining_table','desk':'desk','office table':'desk','computer desk':'desk','coffee table':'coffee_table','coffe table':'coffee_table','night table':'bedside_table','bed table':'bedside_table','nightstand':'bedside_table'}
HER={3:'other_table',4:'coffee_table',5:'not_a_table',10:'coffee_table',14:'bedside_table',16:'not_a_table',17:'not_a_table',19:'coffee_table',21:'bedside_table',24:'not_a_table'}
rows={(r['scene'],r['id']):r for r in map(json.loads,open(root+'/vlm_tc.jsonl'))}
review={s['crop']:s for s in json.load(open(root+'/review_tables.json'))}
targets={}
for k,r in rows.items():
    if r['gt_name'].lower() in SPECIFIC: targets[k]=SPECIFIC[r['gt_name'].lower()]
    elif r['crop'] in review: n=review[r['crop']]['n']; targets[k]=HER.get(n,r['vlm'])
best={}
for f in glob.glob(root+'/*/gt.json'):
    scene=f.split('/')[-2]
    for fr in json.load(open(f)):
        for b in fr['boxes']:
            x0,y0,x1,y1=b['xyxy']; touching=x0<=2 or y0<=2 or x1>=637 or y1>=477
            key=(scene,b['id'])
            if key in targets and (key not in best or (not touching,b['pixels'])>best[key][0]): best[key]=((not touching,b['pixels']),fr['frame'],b['xyxy'])
def b64(img): return base64.b64encode(cv2.imencode('.jpg',img)[1].tobytes()).decode()
def ask(img,labels,framed):
    intro=('In this photo of a home, the piece of furniture inside the red rectangle is table-like. Use the surrounding room to decide which kind it is.' if framed
           else 'This is a cropped photo of a piece of furniture from a home. Which kind is it?')
    p='%s Answer with exactly one of: %s. Reply as JSON {"kind": "<one>"}.'%(intro,', '.join(labels))
    body={'model':MODEL,'stream':False,'think':False,'format':'json','options':{'temperature':0,'num_predict':60,'num_ctx':4096},'messages':[{'role':'user','content':p,'images':[b64(img)]}]}
    try: return json.loads(requests.post(URL,json=body,timeout=300).json()['message']['content']).get('kind')
    except Exception as e: return 'ERROR '+repr(e)[:50]
out=open(root+'/table_variants.jsonl','w')
for (scene,iid),(_,frame,xyxy) in sorted(best.items()):
    img=cv2.imread(f'{root}/{scene}/frames/{frame}'); h,w=img.shape[:2]; x0,y0,x1,y1=[int(v) for v in xyxy]
    framed=img.copy(); cv2.rectangle(framed,(x0,y0),(x1,y1),(0,0,255),3)
    px,py=int(.25*(x1-x0)),int(.25*(y1-y0)); crop=img[max(0,y0-py):min(h,y1+py),max(0,x0-px):min(w,x1+px)]
    out.write(json.dumps({'scene':scene,'id':iid,'truth':targets[(scene,iid)],'crop_other':rows[(scene,iid)]['vlm'],
        'frame_other':ask(framed,WITH_OTHER,True),'crop_noother':ask(crop,NO_OTHER,False),'frame_noother':ask(framed,NO_OTHER,True)})+'\n'); out.flush()
print('VARIANTS_DONE',len(best))
