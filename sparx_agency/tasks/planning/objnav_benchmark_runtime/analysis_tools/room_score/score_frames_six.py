import json,os,sys,time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
HOME=os.path.expanduser('~'); MODEL=sys.argv[1]; TAG=sys.argv[2] if len(sys.argv)>2 else '2b'
KEYS=['chair','bed','potted plant','toilet','tv','couch']
PROMPTS={'A':'Imagine walking into the room shown in this image. How likely is it that you would find a %s in it? Answer yes or no.',
         'B':'A robot is searching for a %s. This image shows a room it could go to next. Is it worth going there to continue the search? Answer yes or no.'}
frames=[]; seen=set()
for b in json.load(open(HOME+'/tc_sweep/buffers.json')):
    for f in b['frames']:
        if f['file'] not in seen: seen.add(f['file']); frames.append(f['file'])
processor=AutoProcessor.from_pretrained(MODEL); processor.tokenizer.padding_side='left'
kw=dict(dtype=torch.float16,device_map='cuda') if TAG=='2b' else dict(dtype=torch.float16,device_map='auto',max_memory={0:'5GiB','cpu':'12GiB'})
model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,**kw).eval()
tok=processor.tokenizer; dev=next(model.parameters()).device
yes=[tok.encode(w,add_special_tokens=False)[0] for w in ('Yes','yes',' Yes',' yes')]; no=[tok.encode(w,add_special_tokens=False)[0] for w in ('No','no',' No',' no')]
for v,P in PROMPTS.items():
    path=f'{HOME}/tc_sweep/vlm_frame_six_{TAG}_{v}.jsonl'
    done={json.loads(l)['file'] for l in open(path)} if os.path.exists(path) else set()
    todo=[f for f in frames if f not in done]; print('variant',v,TAG,'to do',len(todo),flush=True)
    out=open(path,'a')
    for i,f in enumerate(todo):
        t0=time.monotonic()
        try:
            img=Image.open(f'{HOME}/{f}').convert('RGB')
            msgs=[[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':P%k}]}] for k in KEYS]
            inputs=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True).to(dev)
            with torch.no_grad(): lg=model(**inputs).logits[:,-1,:].float()
            lp=torch.log_softmax(lg,dim=-1); py=torch.logsumexp(lp[:,yes],-1); pn=torch.logsumexp(lp[:,no],-1)
            row={'file':f,'p_yes':dict(zip(KEYS,[round(float(x),4) for x in torch.sigmoid(py-pn)]))}
        except Exception as e: row={'file':f,'p_yes':None,'err':repr(e)[:150]}
        row['s']=round(time.monotonic()-t0,2); out.write(json.dumps(row)+'\n'); out.flush()
        if i%50==49: torch.cuda.empty_cache()
print('SCORE_DONE',flush=True)
