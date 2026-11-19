import json,os,sys,time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
HOME=os.path.expanduser('~'); MODEL=sys.argv[1]; LIMIT=int(sys.argv[2]) if len(sys.argv)>2 else 10**9
OUT=HOME+'/tc_sweep/vlm_frame_logprob.jsonl'
KEYS=['chair','bed','potted plant','toilet','tv','couch']
Q='This image shows one view inside a room of a home. Is there a %s somewhere in this room (it may be outside this view)? Answer yes or no.'
frames=[]; seen=set()
for b in json.load(open(HOME+'/tc_sweep/buffers.json')):
    for f in b['frames']:
        if f['file'] not in seen: seen.add(f['file']); frames.append(f['file'])
done={json.loads(l)['file'] for l in open(OUT)} if os.path.exists(OUT) else set()
todo=[f for f in frames if f not in done][:LIMIT]; print('frames total',len(frames),'done',len(done),'to do',len(todo),flush=True)
processor=AutoProcessor.from_pretrained(MODEL); processor.tokenizer.padding_side='left'
model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.float16,device_map='cuda').eval()
tok=processor.tokenizer
yes=[tok.encode(w,add_special_tokens=False)[0] for w in ('Yes','yes',' Yes',' yes')]; no=[tok.encode(w,add_special_tokens=False)[0] for w in ('No','no',' No',' no')]
out=open(OUT,'a')
for i,f in enumerate(todo):
    t0=time.monotonic()
    try:
        img=Image.open(f'{HOME}/{f}').convert('RGB')
        msgs=[[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':Q%k}]}] for k in KEYS]
        inputs=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True).to('cuda')
        with torch.no_grad(): logits=model(**inputs).logits[:,-1,:].float()
        lp=torch.log_softmax(logits,dim=-1); py=torch.logsumexp(lp[:,yes],dim=-1); pn=torch.logsumexp(lp[:,no],dim=-1)
        score=torch.sigmoid(py-pn).cpu().tolist(); row={'file':f,'p_yes':dict(zip(KEYS,[round(s,4) for s in score])),'mass_yes_no':[round(float(torch.exp(py[j])+torch.exp(pn[j])),3) for j in range(len(KEYS))]}
    except Exception as e: row={'file':f,'p_yes':None,'err':repr(e)[:150]}
    row['s']=round(time.monotonic()-t0,2); row['gpu_mb']=int(torch.cuda.max_memory_allocated()/1e6)
    out.write(json.dumps(row)+'\n'); out.flush()
    if i%50==49: torch.cuda.empty_cache()
print('SCORE_DONE',len(todo),flush=True)
