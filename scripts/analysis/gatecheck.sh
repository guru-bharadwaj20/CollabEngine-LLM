#!/usr/bin/env bash
# Evaluate the Phase 1 gate the moment the baseline episodes exist, which is
# well before stage 1 finishes and about ninety minutes before the ablation
# grid starts.
#
# The pipeline runs stage 2 straight after stage 1 with no pause. docs/PLAN.md makes
# a failed gate a stop condition, so without something watching, a corpus that
# has already shown the team contributing nothing still costs nine hours of
# ablation measuring its own noise floor.
#
# This deliberately reports rather than kills. The verdict depends on which
# metric you read and on n=12 episodes, which is too thin for the decision to
# be made by a threshold in a shell script.
set -u

RUN_DIR=runs/qwen3-8b-local
NEED=${1:-12}          # baseline episodes required before judging
OUT=runs/gate.txt

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT"; }

say "waiting for $NEED baseline episodes"
while true; do
  n=$(python - "$RUN_DIR/baseline.jsonl" "$NEED" <<'PY'
import json, sys
path, need = sys.argv[1], int(sys.argv[2])
try:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
except OSError:
    print(0); raise SystemExit
print(sum(1 for r in rows if r["condition"] == "baseline"))
PY
)
  [ "${n:-0}" -ge "$NEED" ] && break
  if [ "$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*collabengine.cli pipeline*' } | Measure-Object).Count" | tr -d '\r ')" = "0" ]; then
    say "pipeline gone with only ${n:-0} baseline episodes; judging on what landed"
    break
  fi
  sleep 120
done

say "=== PHASE 1 GATE ==="
python - "$RUN_DIR/baseline.jsonl" 2>&1 <<'PY' | tee -a "$OUT"
import json, statistics as st, sys
from collections import defaultdict
from collabengine.analysis.scoring import METRICS, rescore
from collabengine.transcripts.store import TranscriptReader
from collabengine.analysis.integrity import is_instrument_failure

by = defaultdict(lambda: defaultdict(list))
for rec in TranscriptReader(sys.argv[1]):
    if is_instrument_failure(rec) or rec.condition not in ("solo", "baseline"):
        continue
    r = rescore(rec)
    for m in METRICS:
        by[m][rec.condition].append(r.overall[m])

if "solo" not in by["fraction"] or "baseline" not in by["fraction"]:
    print("  incomplete: need both arms")
    raise SystemExit

print(f"{'metric':<12}{'solo':>8}{'team':>8}{'gap':>9}{'d':>8}")
verdict = []
for m in METRICS:
    s, t = by[m]["solo"], by[m]["baseline"]
    gap = st.mean(t) - st.mean(s)
    pooled = ((st.pstdev(s) ** 2 + st.pstdev(t) ** 2) / 2) ** 0.5
    d = gap / pooled if pooled else 0.0
    verdict.append(d)
    print(f"{m:<12}{st.mean(s):>8.3f}{st.mean(t):>8.3f}{gap:>+9.3f}{d:>+8.2f}")

best = max(verdict)
n = min(len(by['fraction']['solo']), len(by['fraction']['baseline']))
print(f"\nn={n} per arm; largest effect across metrics d={best:+.2f}")

# Refuse rather than caveat. An earlier version printed the warning below and
# then a verdict anyway, and rendered "gate passes" off a single episode per
# arm -- the integrity filter had removed the rest. A qualified verdict gets
# quoted without its qualification; no verdict does not.
if n < 5:
    print(f"VERDICT: none. {n} usable episodes per arm is too few to read.")
    print("  If the corpus looked larger than this, the integrity filter")
    print("  removed the difference -- check errored turns per condition.")
    raise SystemExit

if best < 0.5:
    print("VERDICT: gate FAILS -- the team is not measurably better than one")
    print("  agent on any metric. An ablation grid here measures the noise")
    print("  floor. Stop stage 2 and reconsider the task, not the difficulty.")
else:
    print("VERDICT: gate passes -- proceed to the ablation grid, and read the")
    print("  drops on the same metric that showed the gap.")
PY
say "=== END GATE ==="
