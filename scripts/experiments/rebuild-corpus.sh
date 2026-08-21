#!/usr/bin/env bash
# Regenerate the fresh-seed `medium` corpus every headline number is read from.
#
# **This is a re-measurement, not a reproduction, and the difference is the
# point.** The corpus behind RESEARCH-LOG 4.19-4.23 is gone: `runs/` was cleaned
# and every *.jsonl is gitignored, so nothing on disk or in git can regenerate
# those exact episodes. Instances are deterministic in (seed, difficulty) and
# the seeds are recorded, so the *instances* come back identically. The
# generations do not: llama.cpp is deterministic per slot but not across
# continuous-batching arrangements (docs/LLAMACPP-SETUP.md), and the weights are
# now a different upload of the same quantisation (see below).
#
# So the honest claim is: same instrument geometry, same seeds, same
# quantisation, independently sampled generations. Numbers that move are
# informative; numbers that move a lot are a finding.
#
# **Weights provenance changed, deliberately.** Every Llama GGUF now comes from
# mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF so that the Q4_K_M instrument and
# the Q8_0 / f16 rungs of the precision ladder share one conversion. A ladder
# assembled from three uploaders would confound precision with whoever ran the
# conversion. `scripts/ops/fetch_models.py` has the reasoning.
#
#   scripts/ops/serve.sh --detach
#   bash scripts/experiments/rebuild-corpus.sh
#
# Resumes on plan id, so a death costs only the unfinished episodes.
set -uo pipefail

LOG=runs/rebuild-corpus.log
CONFIG=${CONFIG:-configs/llamacpp/medium-h3b.yaml}
PY=${PY:-python}

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "rebuilding the fresh-seed medium corpus from $CONFIG"

if ! $PY -u scripts/ops/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
  say "PREFLIGHT FAILED -- nothing after this runs clean"
  exit 1
fi

# Order matters only for how soon a number becomes readable. The gate needs
# baseline and solo, so those go first; solo_long is the matched-budget arm and
# the ablation grid is the most expensive, so they follow.
say "=== 1/2 baseline, solo, solo_long ==="
if ! $PY -u -m collabengine.cli pipeline --config "$CONFIG" \
     --phases baseline,solo,solo_long 2>&1 | tee -a "$LOG"; then
  say "FAILED the three headline arms"
  exit 1
fi
say "GATE IS READABLE -- scripts/analysis/gate_report.py"

say "=== 2/2 live ablation, A1-A4 ==="
if ! $PY -u -m collabengine.cli ablate --config "$CONFIG" \
     --modes live --agents A1,A2,A3,A4 2>&1 | tee -a "$LOG"; then
  say "FAILED the ablation arms; the gate above is still good"
  exit 1
fi

say "ALL DONE"
