import collections,json
R=[json.loads(l) for l in open('/home/daphnaa/tc_sweep/room_more.jsonl')]
V=['baseline','subtypes','tables_sub','cabinets_sub','no_cabinet','no_chair','no_desk','no_table']
by=collections.defaultdict(dict)
for r in R: by[r['n']][r['variant']]=r
clear=[n for n,d in by.items() if d['baseline']['truth'] not in ('unclear','other')]
print('rooms',len(by),'| clear',len(clear),'| unclear',sum(d['baseline']['truth']=='unclear' for d in by.values()),'| other',sum(d['baseline']['truth']=='other' for d in by.values()))
print('\nclear rooms: correct per variant')
for v in V:
    ok=sum(by[n][v]['mapped']==by[n][v]['truth'] for n in clear); none=sum(by[n][v]['mapped']=='none' for n in clear)
    print('  %-14s %2d/%d (%d%%)   unknown/other-type answers: %d'%(v,ok,len(clear),100*ok/len(clear),none))
print('\nrooms whose answer changed vs baseline (clear rooms): variant -> n: baseline -> variant (truth)')
for v in V[1:]:
    ch=[(n,by[n]['baseline']['mapped'],by[n][v]['mapped'],by[n][v]['truth']) for n in clear if by[n][v]['mapped']!=by[n]['baseline']['mapped']]
    print(' ',v,len(ch),'changed; better',sum(c[2]==c[3] and c[1]!=c[3] for c in ch),'worse',sum(c[1]==c[3] and c[2]!=c[3] for c in ch))
    for n,a,b,t in ch: print('     room %2d: %s -> %s (truth %s)'%(n,a,b,t))
print('\nunclear rooms: baseline answers',collections.Counter(by[n]['baseline']['mapped'] for n in by if by[n]['baseline']['truth']=='unclear'))
print('other (basement) rooms:',[(n,by[n]['baseline']['more'],by[n]['subtypes']['more']) for n in by if by[n]['baseline']['truth']=='other'])
print('\nrooms with no objects from GT:',[n for n in by if by[n]['baseline']['n_obj']==0])
