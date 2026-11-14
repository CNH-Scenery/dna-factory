#!/usr/bin/env bash
# One-shot GPU experiment for the frontier curriculum (docs/grpo-frontier-curriculum.md).
#   1. unlock curve: does showing teacher steps turn all-wrong groups into ones with signal?
#   2. control (never a hint) vs MFC, identical otherwise.
# Everything is echoed to the container's stdout as well, so it shows up in the host's logs.
#   bash experiments/frontier/run_all.sh            # from anywhere
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="${MODEL:-dnotitia/Qwen3-0.6B}"
BUDGET="${BUDGET:-150m}"   # wall-clock cap per training run
mkdir -p runs
SINK=/proc/1/fd/1; [ -w "$SINK" ] || SINK=/dev/null
say() { echo "FRONTIER $(date -u +%H:%M:%S) $*" | tee -a runs/all.log "$SINK"; }
pipe() { tee -a runs/all.log "$SINK"; }

say "start; model=$MODEL budget=$BUDGET commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | pipe
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
say "installing dependencies (uv sync)"
if ! uv sync --frozen > runs/uv_sync.log 2>&1; then
  tail -20 runs/uv_sync.log | pipe; say "FAILED: uv sync"; exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate
say "disk after install: $(df -h . | tail -1)"

python utils/build-frontier-grpo-dataset.py --output_dir data/uns128_frontier 2>&1 | grep -v "shards\|Generating\|Downloading" | pipe

say "unlock curve"
python -u experiments/frontier/unlock_curve.py "$MODEL" runs/unlock_curve.json vllm 4096 2>&1 \
  | grep --line-buffered "FRONTIER_UNLOCK\|prompts;\|generation done\|Error\|error" | pipe
say "unlock curve exit=${PIPESTATUS[0]}"

for name in control mfc; do
  say "training $name"
  timeout "$BUDGET" python -u grpo.py --config "experiments/frontier/$name.yaml" --model_name_or_path "$MODEL" 2>&1 \
    | tee "runs/$name.full.log" | python -u experiments/frontier/compact_log.py "$name" | pipe
  say "training $name exit=${PIPESTATUS[0]} (124 = stopped at the time budget)"
done
say "done"
