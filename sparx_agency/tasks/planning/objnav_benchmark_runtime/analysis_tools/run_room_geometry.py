import collections,json,os,sys
sys.path.insert(0,os.path.expanduser('~/GIT/TheAgency'))
os.environ.update(LLM_BACKEND='ollama',LLM_BASE_URL='http://127.0.0.1:11435',LLM_MODEL='qwen2.5:3b-instruct')
from sparx_agency.core.mapping.topology.llm_client import LLMClient,LLMConfig
from sparx_agency.core.mapping.topology.room_classifier import RoomTypeClassifier
root=os.path.expanduser('~/tc_sweep')
STRUCT={'wall','ceiling','floor','door frame','doorframe','window frame','unknown','window','door','ceiling lamp','curtain','painting','picture','photo mount','window glass','shower floor','shower wall','decoration','radiator','parapet','railing','stairs','paper','cable','light switch','shoe','shoes','pillow','box','book','clock','cloth'}
TABLES={'table','coffee table','dining table','kitchen table','night table','bed table','side table','small table','tv table','display table','sink table','coffe table','folded table'}
DESKS={'desk','office table','computer desk'}
def class_of(name):
    n=name.lower()
    if n.endswith(('door','drawer')) or n in STRUCT: return None
    if n in DESKS: return 'desk'
    if n in TABLES: return 'dining table'
    if any(w in n for w in ('cabinet','wardrobe','cupboard','sideboard','dresser','chest of drawers','tv stand','shelf','bookshelf','bookcase')) or n=='closet': return 'cabinet'
    if 'armchair' in n: return 'armchair'
    if 'chair' in n: return 'chair'
    if n=='bed' or (n.endswith(' bed') and 'bedside' not in n): return 'bed'
    if 'plant' in n: return 'potted plant'
    if 'toilet' in n and not any(w in n for w in ('paper','brush')): return 'toilet'
    if any(w in n for w in ('television','monitor')) or n in ('tv','wall tv'): return 'tv'
    if any(w in n for w in ('sofa','couch','loveseat')): return 'couch'
    if any(w in n for w in ('oven','stove','cooktop')): return 'oven'
    if 'sink' in n or 'washbasin' in n: return 'sink'
    if any(w in n for w in ('refrigerator','fridge')): return 'refrigerator'
    if 'bathtub' in n or 'bath tub' in n or n=='bath': return 'bathtub'
    if n.startswith('shower') and 'floor' not in n and 'wall' not in n and 'shelf' not in n: return 'shower'
    if 'stool' in n: return 'stool'
    return None
SUBTYPE_NAME={'dining_table':'dining table','desk':'desk','coffee_table':'coffee table','bedside_table':'bedside table','other_table':'table',
              'kitchen_cabinet':'kitchen cabinet','wardrobe':'wardrobe','bathroom_cabinet':'bathroom cabinet','chest_of_drawers':'chest of drawers','tv_stand':'tv stand','shelf_or_bookcase':'bookshelf','other_cabinet':'cabinet'}
MAP_TO_HER={'kitchen':'kitchen','living_room':'living room','dining_room':'dining room','bedroom':'bedroom','bathroom':'bathroom','office':'office / study','hallway':'hallway / corridor','foyer':'hallway / corridor','storage_closet':'other','laundry_room':'other','garage':'other','balcony':'other'}
sub={(r['scene'],r['id']):r for r in map(json.loads,open(root+'/subtype_all.jsonl'))}
def objects_of(scene,region):
    out=[]
    for o in json.load(open(f'{root}/gt/gtbox_{scene}.json')):
        if o['region']!=region or (o['max'][0]-o['min'][0])>=5 or (o['max'][1]-o['min'][1])>=5: continue
        c=class_of(o['name'])
        if c: out.append({'id':o['id'],'name':o['name'],'cls':c})
    return out
def facts_of(scene,region):
    fl=[o for o in json.load(open(f'{root}/gt/gtbox_{scene}.json')) if o['region']==region and o['name']=='floor']
    if not fl: return None
    area=sum((o['max'][0]-o['min'][0])*(o['max'][1]-o['min'][1]) for o in fl)
    x0=min(o['min'][0] for o in fl); x1=max(o['max'][0] for o in fl); y0=min(o['min'][1] for o in fl); y1=max(o['max'][1] for o in fl)
    a,b=sorted((x1-x0,y1-y0)); return {'area_m2':area,'elongation':b/max(a,0.5)}
def variant_classes(objs,scene,variant):
    classes=[]
    for o in objs:
        c=o['cls']
        if variant=='no_cabinet' and c=='cabinet': continue
        if variant=='no_chair' and c=='chair': continue
        if variant=='no_desk' and c=='desk': continue
        if variant=='no_table' and c=='dining table': continue
        s=sub.get((scene,o['id']))
        if s and ((variant in('subtypes','tables_sub') and s['family']=='table' and c in('dining table','desk')) or (variant in('subtypes','cabinets_sub') and s['family']=='cabinet' and c=='cabinet')):
            if s['subtype'] in ('not_a_table','not_a_cabinet'): continue
            c=SUBTYPE_NAME.get(s['subtype'],c)
        classes.append(c)
    return classes

import math
DOORS={'door','door ','door frame','doorframe','sliding glass door','terrace door'}
def pub(o): return (o['min'][1],o['max'][1],-o['max'][0],-o['min'][0])
def gap(a,b): return math.hypot(max(a[0]-b[1],0,b[0]-a[1]),max(a[2]-b[3],0,b[2]-a[3]))
def door_count(scene,region):
    objs=json.load(open(f'{root}/gt/gtbox_{scene}.json'))
    fl=[o for o in objs if o['region']==region and o['name']=='floor']
    if not fl: return None
    fz=sum((o['min'][2]+o['max'][2])/2 for o in fl)/len(fl); floors=[pub(o) for o in fl]
    cand=[o for o in objs if o['name'].lower() in DOORS and abs(o['min'][2]-fz)<0.6 and any(gap(pub(o),f)<=0.6 for f in floors)]
    centres=[]
    for o in cand:
        b=pub(o); c=((b[0]+b[1])/2,(b[2]+b[3])/2)
        if all(math.dist(c,k)>0.8 for k in centres): centres.append(c)
    return len(centres)
def facts_for(scene,region,variant):
    f=facts_of(scene,region)
    if f is None: return None
    d=door_count(scene,region)
    if variant in ('geo_none',): return None
    if variant=='geo_area': return {'area_m2':f['area_m2']}
    if variant=='geo_elong': return {'elongation':f['elongation']}
    if variant=='geo_area_elong': return f
    return dict(f,doors=d)
VARIANTS=os.environ.get('VARIANTS','geo_none,geo_area,geo_elong,geo_area_elong,geo_all,geo_only_all,geo_none_r2,geo_all_r2').split(',')
client=LLMClient(LLMConfig.from_env()); path=root+'/room_geometry.jsonl'
done={(r['n'],r['variant']) for r in map(json.loads,open(path))} if os.path.exists(path) else set()
out=open(path,'a')
for room in json.load(open(root+'/room_labels.json')):
    objs=objects_of(room['scene'],room['region'])
    for v in (x for x in VARIANTS if (room['n'],x) not in done):
        classes=[] if v=='geo_only_all' else variant_classes(objs,room['scene'],'baseline')
        facts=facts_for(room['scene'],room['region'],v)
        r=RoomTypeClassifier(client,count_sensitive=True).classify(classes,facts=facts,shape_only=True)
        out.write(json.dumps({'n':room['n'],'truth':room['label'],'variant':v,'more':r.label,'mapped':MAP_TO_HER.get(r.label,'none'),'n_obj':len(classes),'facts':facts,'reason':r.reasoning})+'\n'); out.flush()
    print('room',room['n'],flush=True)
print('GEO_DONE')
