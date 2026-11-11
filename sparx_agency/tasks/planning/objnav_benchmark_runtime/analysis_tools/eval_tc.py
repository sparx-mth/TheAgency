import json,collections
rows=[json.loads(l) for l in open('/home/daphnaa/tc_sweep/vlm_tc.jsonl')]
T={'dining table':'dining_table','kitchen table':'dining_table','desk':'desk','office table':'desk','computer desk':'desk','coffee table':'coffee_table','coffe table':'coffee_table','night table':'bedside_table','bed table':'bedside_table','nightstand':'bedside_table','bedside cabinet':'bedside_table',
 'kitchen cabinet':'kitchen_cabinet','kitchen lower cabinet':'kitchen_cabinet','wardrobe':'wardrobe','closet':'wardrobe','bathroom cabinet':'bathroom_cabinet','bath cabinet':'bathroom_cabinet','sink cabinet':'bathroom_cabinet',
 'chest of drawers':'chest_of_drawers','dresser':'chest_of_drawers','tv stand':'tv_stand','shelf':'shelf_or_bookcase','bookshelf':'shelf_or_bookcase'}
print(len(rows),'objects;',sum(r['vlm'].startswith('ERROR') for r in rows),'errors')
spec=[r for r in rows if r['gt_name'].lower() in T]
ok=sum(T[r['gt_name'].lower()]==r['vlm'] for r in spec); print('objects with a specific GT name: %d | VLM matches %d (%d%%)'%(len(spec),ok,100*ok/len(spec)))
for fam in ('table','cabinet'):
    s=[r for r in spec if r['family']==fam]; 
    if s: print(' ',fam,len(s),'->',sum(T[r['gt_name'].lower()]==r['vlm'] for r in s),'right')
print('\nGT -> what the VLM said (specific names only)')
by=collections.defaultdict(collections.Counter)
for r in spec: by[T[r['gt_name'].lower()]][r['vlm']]+=1
for g,c in sorted(by.items(),key=lambda kv:-sum(kv[1].values())): print('  %-18s n=%3d | %s'%(g,sum(c.values()),', '.join('%s %d'%(k,v) for k,v in c.most_common(4))))
gen=[r for r in rows if r['gt_name'].lower() in ('table','cabinet')]
print('\ngeneric GT names (no subtype in GT):',len(gen))
for n in ('table','cabinet'):
    c=collections.Counter(r['vlm'] for r in gen if r['gt_name'].lower()==n); print('  VLM on "%s": %s'%(n,dict(c.most_common(8))))
