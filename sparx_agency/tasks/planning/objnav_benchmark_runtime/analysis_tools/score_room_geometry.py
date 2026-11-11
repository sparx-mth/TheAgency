import collections,json
R=[json.loads(l) for l in open('/home/daphnaa/tc_sweep/room_geometry.jsonl')]
B=[json.loads(l) for l in open('/home/daphnaa/tc_sweep/room_more.jsonl')]
by=collections.defaultdict(dict)
for r in R+[b for b in B if b['variant'] in('baseline','baseline_r2','baseline_r3')]: by[r['n']][r['variant']]=r
hall=lambda n:by[n]['geo_all']['truth']
clear=[n for n,d in by.items() if hall(n) not in('unclear','other') and 'geo_all' in d]
print('clear rooms',len(clear),'of',len(by))
names=[('geo_none','objects only (no geometry)'),('geo_area','objects + area'),('geo_elong','objects + shape'),('geo_area_elong','objects + area + shape (today)'),('geo_all','objects + area + shape + doors'),('geo_only_all','geometry only, no objects'),('geo_none_r2','objects only, repeat'),('geo_all_r2','all geometry, repeat'),('baseline','today, earlier run'),('baseline_r2','today, earlier run 2')]
for k,name in names:
    ns=[n for n in clear if k in by[n]]
    if len(ns)<len(clear): continue
    ok=sum(by[n][k]['mapped']==by[n][k]['truth'] for n in clear); print('  %-34s %2d/%d (%d%%)'%(name,ok,len(clear),100*ok/len(clear)))
hallways=[n for n in clear if hall(n)=='hallway / corridor']
print('\nhallway rooms (%d): correct per variant'%len(hallways))
for k,name in names:
    if all(k in by[n] for n in hallways): print('  %-34s %d/%d'%(name,sum(by[n][k]['mapped']=='hallway / corridor' for n in hallways),len(hallways)))
print('\nhallway rooms: facts and answers')
for n in hallways: print('  room %2d facts %s | none: %s | all: %s | only geometry: %s'%(n,{k:round(v,1) for k,v in (by[n]['geo_all']['facts'] or {}).items()},by[n]['geo_none']['mapped'],by[n]['geo_all']['mapped'],by[n]['geo_only_all']['mapped']))
print('\nnon-hallway rooms called hallway:',{k:sum(by[n][k]['mapped']=='hallway / corridor' for n in clear if hall(n)!='hallway / corridor') for k,_ in names if all(k in by[n] for n in clear)})
