import base64,collections,glob,json,os,random
import cv2,numpy as np
root=os.path.expanduser('~/tc_sweep'); LABELS=['kitchen','living room','dining room','bedroom','bathroom','office / study','hallway / corridor','other','unclear']
def region_finder(scene):
    objs=json.load(open(f'{root}/gt/gtbox_{scene}.json')); floors=[]
    for o in objs:
        if o['name']=='floor': floors.append((o['min'][1],o['max'][1],-o['max'][0],-o['min'][0],(o['min'][2]+o['max'][2])/2,o['region']))
    area=collections.Counter()
    for x0,x1,y0,y1,z,r in floors: area[r]+=(x1-x0)*(y1-y0)
    def find(x,y,z):
        best=None
        for x0,x1,y0,y1,fz,r in floors:
            if abs(fz-z)>0.8: continue
            d=np.hypot(max(x0-x,0,x-x1),max(y0-y,0,y-y1))
            if d<=0.2 and (best is None or d<best[0]): best=(d,r)
        return best[1] if best else None
    return find,area
cands=[]
for scene in sorted(d for d in os.listdir(root) if d[:5].isdigit() and os.path.isdir(f'{root}/{d}')):
    find,area=region_finder(scene); by=collections.defaultdict(list)
    for fr in json.load(open(f'{root}/{scene}/gt.json')):
        r=find(fr['x'],fr['y'],fr['z'])
        if r is not None: by[r].append(fr)
    for r,frs in by.items():
        if len({(f['x'],f['y']) for f in frs})>=1 and len(frs)>=3 and area[r]>=4: cands.append((scene,r,frs))
rng=random.Random(11); rng.shuffle(cands); per=collections.Counter(); chosen=[]
for scene,r,frs in cands:
    if per[scene]<5 and len(chosen)<40: chosen.append((scene,r,frs)); per[scene]+=1
rng.shuffle(chosen); meta=[]; cards=[]
for n,(scene,r,frs) in enumerate(chosen,1):
    rng.shuffle(frs); pick=[]; yaws=set()
    for f in frs:
        if round(f['yaw'],2) not in yaws or len(frs)<=4: pick.append(f); yaws.add(round(f['yaw'],2))
        if len(pick)==4: break
    imgs=[]
    for f in pick:
        im=cv2.resize(cv2.imread(f"{root}/{scene}/frames/{f['frame']}"),(320,240)); imgs.append(base64.b64encode(cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,72])[1].tobytes()).decode())
    meta.append({'n':n,'scene':scene,'region':r,'frames':[f['frame'] for f in pick]})
    cards.append('<div class="room"><h3>Room %d</h3><div class="imgs">%s</div><select id="r%d"><option value="">-- choose --</option>%s</select></div>'%(n,''.join('<img src="data:image/jpeg;base64,%s">'%i for i in imgs),n,''.join('<option>%s</option>'%l for l in LABELS)))
json.dump(meta,open(f'{root}/label_rooms_key.json','w'))
html='''<!doctype html><meta charset="utf-8"><title>Room labels</title><style>body{font:15px sans-serif;margin:16px;max-width:1400px}.room{border:1px solid #bbb;margin:12px 0;padding:8px;border-radius:6px}.imgs img{margin:2px;width:24%%;min-width:200px}select{font-size:16px}button{font-size:16px;padding:8px 14px;margin:8px 0}</style>
<h2>What room is this? (%d rooms, 4 views each)</h2><p>Choose the room type by what the views show. "unclear" if you cannot tell. Then press the button and paste the text in the chat.</p>
<button onclick="out()">Copy answers</button> <span id="msg"></span>%s<button onclick="out()">Copy answers</button><pre id="o"></pre>
<script>function out(){let t=[];for(let i=1;i<=%d;i++){t.push(i+": "+(document.getElementById("r"+i).value||"-"))}const s=t.join("\\n");document.getElementById("o").textContent=s;navigator.clipboard&&navigator.clipboard.writeText(s).then(()=>document.getElementById("msg").textContent="copied")}</script>'''%(len(cards),''.join(cards),len(cards))
open(f'{root}/label_rooms.html','w').write(html); print(len(cards),'rooms','per scene',dict(per),'size MB',round(len(html)/1e6,1))
