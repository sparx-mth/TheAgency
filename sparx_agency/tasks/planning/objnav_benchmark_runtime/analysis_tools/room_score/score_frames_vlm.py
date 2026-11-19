import json,os,re,sys,time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
HOME=os.path.expanduser('~'); MODEL=sys.argv[1]; LIMIT=int(sys.argv[2]) if len(sys.argv)>2 else 10**9
OUT=HOME+'/tc_sweep/vlm_frame_scores.jsonl'
KEYS=['chair','bed','potted plant','toilet','tv','couch']
PROMPT=('This image shows one view inside a room of a home. For each object below, estimate the probability from 0 to 100 that the object is '
        'somewhere in this room (it may be outside this view). Reply with JSON only, with these keys: %s.'%', '.join('"%s"'%k for k in KEYS))
frames=[]; seen=set()
for b in json.load(open(HOME+'/tc_sweep/buffers.json')):
    for f in b['frames']:
        if f['file'] not in seen: seen.add(f['file']); frames.append(f['file'])
done={json.loads(l)['file'] for l in open(OUT)} if os.path.exists(OUT) else set()
todo=[f for f in frames if f not in done][:LIMIT]
print('frames total',len(frames),'done',len(done),'to do',len(todo),flush=True)
processor=AutoProcessor.from_pretrained(MODEL)
model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.float16,device_map='cuda').eval()
out=open(OUT,'a')
for i,f in enumerate(todo):
    t0=time.monotonic()
    try:
        img=Image.open(f'{HOME}/{f}').convert('RGB').resize((640,480),Image.BICUBIC)
        msgs=[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':PROMPT}]}]
        inputs=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt').to('cuda')
        with torch.no_grad(): ids=model.generate(**inputs,max_new_tokens=80,do_sample=False)
        text=processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
        m=re.search(r'\{.*\}',text,re.S); d=json.loads(m.group(0)) if m else {}
        scores={k:(float(d[k]) if k in d and str(d[k]).replace('.','',1).isdigit() else None) for k in KEYS}
        row={'file':f,'scores':scores,'raw':text[:200]}
    except Exception as e: row={'file':f,'scores':None,'raw':'ERROR '+repr(e)[:120]}
    row['s']=round(time.monotonic()-t0,2); row['gpu_mb']=int(torch.cuda.max_memory_allocated()/1e6)
    out.write(json.dumps(row)+'\n'); out.flush()
    if i%50==49: torch.cuda.empty_cache()
print('SCORE_DONE',len(todo),flush=True)
