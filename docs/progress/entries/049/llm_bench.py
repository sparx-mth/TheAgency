"""Throughput of the resident Ollama under N concurrent requests of a representative objnav size.

python llm_bench.py --url http://127.0.0.1:11435 --model qwen2.5:14b-instruct --concurrency 1 4 8 --requests 8
"""
import argparse, json, os, statistics, threading, time, urllib.request
FORMAT_JSON = os.environ.get("FORMAT_JSON") == "1"

ROOM_LINE = ("- room R{i}: type={t}, size={s} m2, doors=2, frontier={f}, objects: {o}; "
             "visited={v}, last seen at action {a}\n")
TYPES = ["bedroom", "living_room", "kitchen", "hallway", "bathroom", "office", "dining_room", "unknown"]
OBJS = ["bed, nightstand, lamp", "sofa, coffee table, tv", "sink, counter, cabinet", "none", "toilet, towel",
        "desk, chair, monitor", "table, 4 chairs", "picture, plant"]


def prompt(rooms):
    head = ("You are the node oracle of an object-search agent in a Matterport house. The target is a CHAIR. "
            "WHERE IT LIVES: chairs stand in dining rooms, offices, living rooms and bedrooms; rarely in bathrooms. "
            "HOUSE: 2 storeys seen, rooms found: %d, reachable frontier left. The agent stands in R0.\n" % rooms)
    body = "".join(ROOM_LINE.format(i=i, t=TYPES[i % 8], s=8 + 3 * (i % 5), f="yes" if i % 3 else "no",
                                    o=OBJS[i % 8], v="yes" if i % 4 == 0 else "no", a=37 * i) for i in range(rooms))
    tail = ("For EVERY room above, give the probability (0-1) that the target is in it and one short reason. "
            "Then name the room to search next. Answer as JSON: {\"rooms\": {\"R0\": {\"p\": 0.1, \"why\": \"...\"}, ...}, "
            "\"next\": \"R3\", \"reading\": \"one sentence\"}. Be thorough.")
    return head + body + tail


def one(url, model, text, num_ctx, num_predict, out):
    payload = {"model": model, "stream": False, "keep_alive": "30m",
               "messages": [{"role": "user", "content": text}],
               "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0, "seed": 17}}
    if FORMAT_JSON:
        payload["format"] = "json"
    t0 = time.time()
    req = urllib.request.Request(url + "/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    out.append({"wall": time.time() - t0, "prompt_tokens": d["prompt_eval_count"], "eval_tokens": d["eval_count"],
                "prompt_s": d["prompt_eval_duration"] / 1e9, "eval_s": d["eval_duration"] / 1e9,
                "total_s": d["total_duration"] / 1e9})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="qwen2.5:14b-instruct")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--rooms", type=int, default=24)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--num-predict", type=int, default=400)
    a = ap.parse_args()
    text = prompt(a.rooms)
    out = []
    one(a.url, a.model, "Reply OK.", a.num_ctx, 2, out)        # warm
    for c in a.concurrency:
        results, threads = [], []
        t0 = time.time()
        for k in range(a.requests):
            # vary the prompt a little so the prefix cache does not answer for the prefill
            txt = text.replace("The agent stands in R0", "The agent stands in R%d after %d actions" % (k % a.rooms, 11 * k))
            th = threading.Thread(target=one, args=(a.url, a.model, txt, a.num_ctx, a.num_predict, results))
            threads.append(th)
            while sum(t.is_alive() for t in threads) >= c:
                time.sleep(0.05)
            th.start()
        for th in threads:
            th.join()
        wall = time.time() - t0
        pt = sum(r["prompt_tokens"] for r in results); et = sum(r["eval_tokens"] for r in results)
        print(json.dumps({"concurrency": c, "requests": a.requests, "wall_s": round(wall, 1),
                          "req_per_s": round(a.requests / wall, 3),
                          "mean_prompt_tokens": pt // a.requests, "mean_eval_tokens": et // a.requests,
                          "mean_wall_per_req_s": round(statistics.fmean(r["wall"] for r in results), 1),
                          "aggregate_eval_tok_per_s": round(et / wall, 1),
                          "single_eval_tok_per_s": round(statistics.fmean(r["eval_tokens"] / r["eval_s"] for r in results), 1),
                          "single_prefill_tok_per_s": round(statistics.fmean(r["prompt_tokens"] / r["prompt_s"] for r in results), 0)}))


if __name__ == "__main__":
    main()
