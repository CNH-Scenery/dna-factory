"""Turn the trainer's per-step metrics dict into one short line per step.

usage: python -u grpo.py ... 2>&1 | python experiments/frontier/compact_log.py NAME
Lines that are not a metrics dict pass through (cut to 300 characters); progress bars are dropped.
"""

import ast
import re
import sys

NAME = sys.argv[1]
KEYS = {
    "reward": "reward",
    "frontier/heldout_solve_rate": "heldout",
    "frontier/solve_rate_unguided": "unguided",
    "frontier/solve_rate_guided": "guided",
    "frontier/unguided_mass": "mass0",
    "frontier/hint_ratio": "hint",
    "frontier/mean_help": "help",
    "frontier/unlocked_frac": "unlocked",
    "frontier/stalled_frac": "stalled",
    "frontier/unhelped_frac": "unhelped",
    "completions/mean_length": "len",
    "completions/clipped_ratio": "clipped",
}
step = 0
for raw in sys.stdin:
    line = re.sub(r"\x1b\[[0-9;]*m", "", raw).rstrip()
    found = re.search(r"\{'[a-z_/]+': .*\}", line)
    if found:
        try:
            logs = ast.literal_eval(found.group(0))
        except ValueError, SyntaxError:
            logs = None
        if isinstance(logs, dict) and "reward" in logs:
            step += 1
            parts = [
                f"{short}={logs[key]}" for key, short in KEYS.items() if key in logs
            ]
            print(f"FRONTIER_STEP {NAME} step={step} " + " ".join(parts), flush=True)
            continue
    if "it/s]" in line or "s/it]" in line or not line.strip():
        continue
    print(line[-300:], flush=True)
