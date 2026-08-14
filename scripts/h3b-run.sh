#!/usr/bin/env bash
# H3b: does the ablation instrument measure the agent, or the configuration?
#
# Two arms, same three agents, same seeds, same instrument. One is a four-agent
# team with A4 excluded; the other is a native three-agent team. If they differ,
# then every live-ablation drop this project has reported is partly a property
# of the measurement, and the confirmatory grid would be measuring the wrong
# thing. PREREG-phase3 Amendment 1, H3b.
#
# Order: baseline first, because `ablate` derives its plans from recorded
# baseline episodes and there are none at these seeds yet.
#
# Everything resumes on plan id, so a death costs only the unfinished episodes.
# Start the server first, or let scripts/overnight-watch.sh do it.
set -uo pipefail

LOG=runs/h3b-run.log
CONFIG=configs/llamacpp-medium-h3b.yaml
EPISODES=${EPISODES:-150}
AGENT=${AGENT:-A4}   # the only agent whose roster the capacity arm matches

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "H3b on $CONFIG, $EPISODES episodes, ablating $AGENT"

if ! python -u scripts/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
  say "PREFLIGHT FAILED -- nothing after this runs clean"
  exit 1
fi

say "=== 1/2 baseline, $EPISODES episodes at seeds 1000-$((1000 + EPISODES - 1)) ==="
if ! python -u -m collabengine.cli pipeline --config "$CONFIG" \
     --phases baseline --episodes "$EPISODES" 2>&1 | tee -a "$LOG"; then
  say "FAILED baseline"
  exit 1
fi

say "=== 2/2 live:$AGENT + capacity, the roster-matched pair ==="
if ! python -u -m collabengine.cli ablate --config "$CONFIG" \
     --modes live,capacity --agents "$AGENT" 2>&1 | tee -a "$LOG"; then
  say "FAILED ablation"
  exit 1
fi

say "ALL DONE -- python -m collabengine.cli analyze --config $CONFIG"
