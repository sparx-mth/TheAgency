import base64,glob,json,os,time
import cv2,requests
root=os.path.expanduser('~/tc_sweep'); URL='http://127.0.0.1:11435/api/chat'; MODEL='qwen3-vl:4b-instruct'
WITH_OTHER=['kitchen_cabinet','wardrobe','bathroom_cabinet','chest_of_drawers','tv_stand','shelf_or_bookcase','other_cabinet','not_a_cabinet']
NO_OTHER=['kitchen_cabinet','wardrobe','bathroom_cabinet','chest_of_drawers','tv_stand','shelf_or_bookcase','not_a_cabinet']
SPECIFIC={'kitchen cabinet':'kitchen_cabinet','kitchen lower cabinet':'kitchen_cabinet','wardrobe':'wardrobe','closet':'wardrobe','bathroom cabinet':'bathroom_cabinet','bath cabinet':'bathroom_cabinet','sink cabinet':'bathroom_cabinet','chest of drawers':'chest_of_drawers','dresser':'chest_of_drawers','tv stand':'tv_stand','shelf':'shelf_or_bookcase','bookshelf':'shelf_or_bookcase'}
rows={(r['scene'],r['id']):r for r in map(json.loads,open(root+'/vlm_tc.jsonl'))}
targets={}
for k,r in rows.items():
    if r['gt_name'].lower() in SPECIFIC: targets[k]=SPECIFIC[r['gt_name'].lower()]
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
    intro=('In this photo of a home, the piece of furniture inside the red rectangle is cabinet-like. Use the surrounding room to decide which kind it is.' if framed
           else 'This is a cropped photo of a piece of furniture from a home. Which kind is it?')
    p='%s Answer with exactly one of: %s. Reply as JSON {"kind": "<one>"}.'%(intro,', '.join(labels))
    body={'model':MODEL,'stream':False,'think':False,'format':'json','options':{'temperature':0,'num_predict':60,'num_ctx':4096},'messages':[{'role':'user','content':p,'images':[b64(img)]}]}
    try: return json.loads(requests.post(URL,json=body,timeout=300).json()['message']['content']).get('kind')
    except Exception as e: return 'ERROR '+repr(e)[:50]
out=open(root+'/cabinet_variants.jsonl','w')
for (scene,iid),(_,frame,xyxy) in sorted(best.items()):
    img=cv2.imread(f'{root}/{scene}/frames/{frame}'); h,w=img.shape[:2]; x0,y0,x1,y1=[int(v) for v in xyxy]
    framed=img.copy(); cv2.rectangle(framed,(x0,y0),(x1,y1),(0,0,255),3)
    px,py=int(.25*(x1-x0)),int(.25*(y1-y0)); crop=img[max(0,y0-py):min(h,y1+py),max(0,x0-px):min(w,x1+px)]
    out.write(json.dumps({'scene':scene,'id':iid,'truth':targets[(scene,iid)],'crop_other':rows[(scene,iid)]['vlm'],
        'frame_other':ask(framed,WITH_OTHER,True),'crop_noother':ask(crop,NO_OTHER,False),'frame_noother':ask(framed,NO_OTHER,True)})+'\n'); out.flush()
print('VARIANTS_DONE',len(best))
