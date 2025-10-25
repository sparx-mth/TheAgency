#!/usr/bin/env python3
import json, os, time, torch
from sentence_transformers import SentenceTransformer, util

MODEL = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
ROOMS, REQ, OUT = "data/unified_rooms.json", "data/object_request.json", "data/object_location.json"
THRESH = 0.55

obj2room, objs, embs = {}, [], None

def update_objs():
    global obj2room, objs, embs
    try:
        data = json.load(open(ROOMS))
        all_pairs = {o["type"]: r for r, d in data["rooms"].items() for o in d["objects"] if "type" in o}
        new = [o for o in all_pairs if o not in obj2room]
        if new:
            e = MODEL.encode(new, convert_to_tensor=True)
            embs = torch.cat([embs, e]) if embs is not None else e
            for o in new: obj2room[o] = all_pairs[o]
            objs = list(obj2room.keys())
    except: pass

def main():
    last_rooms = last_req = 0
    update_objs()
    while True:
        try:
            # detect updated rooms file
            if os.path.exists(ROOMS):
                t = os.path.getmtime(ROOMS)
                if t > last_rooms:
                    update_objs(); last_rooms = t

            # detect new request
            if os.path.exists(REQ):
                t = os.path.getmtime(REQ)
                if t > last_req:
                    req = json.load(open(REQ)).get("task", "").strip()
                    if req and embs is not None:
                        t0 = time.time()
                        q = MODEL.encode(req, convert_to_tensor=True)
                        sims = util.cos_sim(q, embs)[0]
                        i = int(torch.argmax(sims))
                        if sims[i] >= THRESH:
                            obj = objs[i]; room = obj2room[obj]
                        else:
                            obj = room = "none"
                        json.dump({"room": room, "object": obj}, open(OUT, "w"), indent=2)
                        print(f"{room}: {obj} ({time.time()-t0:.3f}s)")
                    last_req = t
            time.sleep(0.5)
        except KeyboardInterrupt:
            break
        except:
            time.sleep(1)

if __name__ == "__main__":
    main()
