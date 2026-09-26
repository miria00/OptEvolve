"""Ask the proposer for a recipe, score against the decisive moves."""
import argparse, json, os, re, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recipes import PROBLEMS, FRAMEWORK, SYS

# The context actually in hand when each recipe was produced: the paper's own
# methods text, verbatim from 3-methods.tex and appendix-methods.tex rather
# than a paraphrase written after the fact.
_pc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_context.txt")
PAPER = open(_pc).read() if os.path.exists(_pc) else ""

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--out", default="")
a = ap.parse_args()

# 2026-09-25: two launches raced on the same --out and held the same log on
# fd 1 at independent offsets, clobbering each other's bytes mid-line.  The
# run had to be discarded.  Refuse to start if another holds this lock.
if a.out:
    import fcntl
    _lock = open(a.out + ".lock", "w")
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("refusing to start: another run already holds %s.lock" % a.out)
    _lock.write(str(os.getpid()) + "\n")
    _lock.flush()


def ask(prompt, seed):
    body = {"model": a.model, "temperature": a.temperature, "max_tokens": 1600,
            "seed": seed, "messages": [{"role": "system", "content": SYS},
                                       {"role": "user", "content": prompt}],
            # Qwen3.8 burns the entire budget in its reasoning channel and
            # returns finish_reason=length with BOTH content and reasoning
            # empty, at 600, 2000, 4000 and 16000 tokens alike.  Neither
            # reasoning_effort=low nor a /no_think suffix helps.  Only
            # enable_thinking=False produces an answer.  Measured before use,
            # because an empty reply scores zero on every key and would read
            # as "the model failed to discover" rather than "the model was
            # never asked".
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=240) as r:
        d = json.load(r)
    m = d["choices"][0]["message"]
    return (m.get("content") or m.get("reasoning_content") or "")


def score(text, keys):
    t = text.lower()
    return {k: any(s in t for s in forms) for k, forms in keys.items()}


CONDS = ("cold", "failure", "paper")
rows, agg = [], {}
for pname, P in PROBLEMS.items():
    for cond in CONDS:
        base = P["failure" if cond == "paper" else cond].replace("[cold]", P["cold"])
        if cond == "paper":
            prompt = ("Here is the method this solver is built inside. It is the "
                      "search space available to you.\n\n" + PAPER +
                      "\n\n----\n\n" + base)
        else:
            prompt = FRAMEWORK + "\n\n" + base
        hits = {k: 0 for k in P["keys"]}
        for i in range(a.n):
            try:
                txt = ask(prompt, 1000 + i)
            except Exception as e:
                txt = ""
            sc = score(txt, P["keys"])
            for k, v in sc.items():
                hits[k] += int(v)
            rows.append({"problem": pname, "cond": cond, "i": i,
                         "scored": sc, "n_moves": sum(sc.values()),
                         "text": txt[:1500]})
        agg[(pname, cond)] = hits
        print("%-22s %-8s  " % (pname, cond) +
              "  ".join("%s %d/%d" % (k.split("/")[0][:26], v, a.n)
                        for k, v in hits.items()), flush=True)

print("\n%-22s %-8s %s" % ("problem", "cond", "mean decisive moves named (of %d)"
                           % max(len(P["keys"]) for P in PROBLEMS.values())))
for pname, P in PROBLEMS.items():
    for cond in CONDS:
        sel = [r for r in rows if r["problem"] == pname and r["cond"] == cond]
        m = sum(r["n_moves"] for r in sel) / max(len(sel), 1)
        full = sum(1 for r in sel if r["n_moves"] == len(P["keys"]))
        print("%-22s %-8s %.2f    all-%d in %d/%d samples"
              % (pname, cond, m, len(P["keys"]), full, len(sel)))
if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
