import gzip,json,os,random,sys
root=os.path.expanduser('~/datasets_v01/objectnav/hm3d/v1/train/content'); per=int(sys.argv[2]); scenes=[s[0] for s in json.load(open(sys.argv[1]))]
out={}
for scene in scenes:
    stem=scene.split('-',1)[1]; d=json.load(gzip.open(f'{root}/{stem}.json.gz')); G={}
    for v in d['goals_by_category'].values():
        for g in v: G[g['object_id']]=g['position'][1]
    eligible=[i for i,e in enumerate(d['episodes']) if e['info']['closest_goal_object_id'] in G and abs(G[e['info']['closest_goal_object_id']]-e['start_position'][1])<1.0 and 3.0<=e['info']['geodesic_distance']<=25]
    pick=sorted(random.Random(17).sample(eligible,per)); out[scene]=['%s/%06d'%(scene,i) for i in pick]
    print(scene,len(d['episodes']),'episodes,',len(eligible),'single-floor eligible;',[d['episodes'][i]['object_category'] for i in pick])
json.dump(out,open(os.path.expanduser('~/tc_sweep/train_episodes.json'),'w'),indent=1)
