"""Unlock curve: pass@8 of the base model on the 128 problems at guidance rho in {0,.25,.5,.75,1}.

Prompts are built exactly as the trainer builds them (render_hint / add_hint, DEFAULT_HINT_TEMPLATE).
usage (from the repo root): python experiments/frontier/unlock_curve.py MODEL OUT.json [vllm|hf|stub] [max_new_tokens] [limit_problems]
`stub` fakes the generations (right answer in half of the hinted rollouts) to self-test the scoring.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets import load_from_disk
from transformers import AutoTokenizer

from dna_factory.frontier_curriculum import (
    DEFAULT_HINT_TEMPLATE,
    add_hint,
    as_steps,
    render_hint,
)

MODEL, OUT = sys.argv[1], sys.argv[2]
BACKEND = sys.argv[3] if len(sys.argv) > 3 else "vllm"
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else None
N, RHOS = 8, [0.0, 0.25, 0.5, 0.75, 1.0]

ds = load_from_disk(str(ROOT / "data" / "uns128_frontier"))["train"]
if LIMIT:
    ds = ds.select(range(LIMIT))
tok = AutoTokenizer.from_pretrained(MODEL)
texts, meta = [], []
for i, row in enumerate(ds):
    steps = as_steps(row["steps"])
    for rho in RHOS:
        g = round(rho * len(steps))
        p = (
            row["prompt"]
            if g == 0
            else add_hint(row["prompt"], render_hint(steps[:g], DEFAULT_HINT_TEMPLATE))
        )
        texts.append(
            tok.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
        )
        meta.append(
            {
                "problem": i,
                "rho": rho,
                "g": g,
                "n_steps": len(steps),
                "solution": row["solution"],
                "prompt": p,
            }
        )
plen = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
print(
    f"{len(texts)} prompts; prompt tokens max {max(plen)} mean {sum(plen) / len(plen):.0f}; sample generation prompt tail: {texts[0][-80:]!r}",
    flush=True,
)

t0 = time.time()
gens = []  # per prompt: list of (text, n_tokens)
if BACKEND == "vllm":
    from vllm import LLM, SamplingParams

    llm = LLM(
        MODEL,
        max_model_len=max(plen) + MAXTOK + 16,
        gpu_memory_utilization=0.88,
        seed=0,
    )
    outs = llm.generate(
        texts,
        SamplingParams(n=N, temperature=1.0, top_p=1.0, max_tokens=MAXTOK, seed=0),
    )
    for o in outs:
        gens.append([(c.text, len(c.token_ids)) for c in o.outputs])
elif BACKEND == "stub":
    for m in meta:
        right = "<think>ok</think>\nThe answer is " + m["solution"]
        wrong = "<think>hm</think>\n\\boxed{123456789}"
        gens.append(
            [(right if (m["g"] and k % 2 == 0) else wrong, 10) for k in range(N)]
        )
else:
    import torch
    from transformers import AutoModelForCausalLM

    model = (
        AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    )
    tok.padding_side = "left"
    B = 8
    for s in range(0, len(texts), B):
        enc = tok(
            texts[s : s + B],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                max_new_tokens=MAXTOK,
                num_return_sequences=N,
            )
        out = out[:, enc["input_ids"].shape[1] :].view(len(texts[s : s + B]), N, -1)
        for row_out in out:
            cur = []
            for ids in row_out:
                ids = [t for t in ids.tolist() if t != tok.pad_token_id]
                cur.append((tok.decode(ids, skip_special_tokens=True), len(ids)))
            gens.append(cur)
        print(
            f"  generated {s + B}/{len(texts)} in {time.time() - t0:.0f}s", flush=True
        )
print(f"generation done in {time.time() - t0:.0f}s", flush=True)

from dna_factory.rewards import safe_accuracy_reward, strict_accuracy_reward

records = []
for m, g in zip(meta, gens):
    comps = [[{"role": "assistant", "content": t}] for t, _ in g]
    sol = [m["solution"]] * len(g)
    ids = [[0] * n for _, n in g]
    acc = safe_accuracy_reward(
        prompts=[m["prompt"]] * len(g), completions=comps, solution=sol
    )
    try:
        strict = strict_accuracy_reward(
            prompts=[m["prompt"]] * len(g),
            completions=comps,
            completion_ids=ids,
            solution=sol,
        )
    except Exception as e:  # noqa: BLE001
        strict = [None] * len(g)
        print("strict failed:", type(e).__name__, e, flush=True)
    records.append(
        {
            **{k: m[k] for k in ("problem", "rho", "g", "n_steps", "solution")},
            "acc": acc,
            "strict": strict,
            "tokens": [n for _, n in g],
            "closed_think": [t.count("</think>") == 1 for t, _ in g],
            "sample": g[0][0][-300:],
        }
    )
Path(OUT).parent.mkdir(parents=True, exist_ok=True)
json.dump(
    {
        "model": MODEL,
        "backend": BACKEND,
        "max_new_tokens": MAXTOK,
        "n": N,
        "records": records,
    },
    open(OUT, "w"),
)


def table(key):
    print(f"FRONTIER_UNLOCK [{key}] rho pass@8 mean_acc scored_rollouts")
    for rho in RHOS:
        rs = [r for r in records if r["rho"] == rho]
        vals = [[v for v in r[key] if v is not None] for r in rs]
        p8 = sum(any(v >= 1.0 for v in vs) for vs in vals) / len(rs)
        flat = [v for vs in vals for v in vs]
        print(
            f"FRONTIER_UNLOCK [{key}] {rho:4.2f} {100 * p8:6.1f}% {100 * sum(flat) / max(len(flat), 1):7.1f}% {len(flat)}",
            flush=True,
        )


table("acc")
table("strict")
tk = [n for r in records for n in r["tokens"]]
print(
    f"FRONTIER_UNLOCK completion tokens: mean {sum(tk) / len(tk):.0f}, at cap ({MAXTOK}): {100 * sum(n >= MAXTOK for n in tk) / len(tk):.1f}%; think closed once: {100 * sum(c for r in records for c in r['closed_think']) / len(tk):.1f}%"
)
