import json,os,sys,time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
HOME=os.path.expanduser('~'); MODEL=sys.argv[1]
PROMPTS={'A':'Imagine walking into the room shown in this image. How likely is it that you would find a %s in it? Answer yes or no.',
         'B':'A robot is searching for a %s. This image shows a room it could go to next. Is it worth going there to continue the search? Answer yes or no.'}
target_of={}
for b in json.load(open(HOME+'/tc_sweep/buffers.json')):
    for f in b['frames']: target_of[f['file']]=b['target']['query']
frames=sorted(target_of)
processor=AutoProcessor.from_pretrained(MODEL); model=Qwen3VLForConditionalGeneration.from_pretrained(MODEL,dtype=torch.float16,device_map='cuda').eval()
tok=processor.tokenizer
yes=[tok.encode(w,add_special_tokens=False)[0] for w in ('Yes','yes',' Yes',' yes')]; no=[tok.encode(w,add_special_tokens=False)[0] for w in ('No','no',' No',' no')]
for v,P in PROMPTS.items():
    path=f'{HOME}/tc_sweep/vlm_frame_target_{v}.jsonl'
    done={json.loads(l)['file'] for l in open(path)} if os.path.exists(path) else set()
    todo=[f for f in frames if f not in done]; print('variant',v,'frames',len(frames),'to do',len(todo),flush=True)
    out=open(path,'a')
    for i,f in enumerate(todo):
        t0=time.monotonic()
        try:
            img=Image.open(f'{HOME}/{f}').convert('RGB')
            msgs=[[{'role':'user','content':[{'type':'image','image':img},{'type':'text','text':P%target_of[f]}]}]]
            inputs=processor.apply_chat_template(msgs,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True).to('cuda')
            with torch.no_grad(): lg=model(**inputs).logits[0,-1,:].float()
            lp=torch.log_softmax(lg,dim=-1); py=torch.logsumexp(lp[yes],0); pn=torch.logsumexp(lp[no],0)
            row={'file':f,'target':target_of[f],'p_yes':round(float(torch.sigmoid(py-pn)),4),'mass':round(float(torch.exp(py)+torch.exp(pn)),3)}
        except Exception as e: row={'file':f,'p_yes':None,'err':repr(e)[:150]}
        row['s']=round(time.monotonic()-t0,2); out.write(json.dumps(row)+'\n'); out.flush()
        if i%50==49: torch.cuda.empty_cache()
print('SCORE_DONE',flush=True)
