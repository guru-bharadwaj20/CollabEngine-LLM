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
# Start the server first, or let scripts/ops/overnight-watch.sh do it.
set -uo pipefail

LOG=runs/h3b-run.log
CONFIG=configs/llamacpp/medium-h3b.yaml
EPISODES=${EPISODES:-150}
AGENT=${AGENT:-A4}   # the only agent whose roster the capacity arm matches

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "H3b on $CONFIG, $EPISODES episodes, ablating $AGENT"

if ! python -u scripts/ops/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
  say "PREFLIGHT FAILED -- nothing after this runs clean"
  exit 1
fi

# Ablation FIRST, baseline second, and the order is the point.
#
# H3b is a two-arm contrast -- live:A4 against capacity:3, same roster, same
# seeds. It does not use the baseline at all; the baseline only expresses those
# two as drops. Running it first put 150 twelve-turn episodes, about 40% of the
# total work, on the critical path of a question that does not ask them. On a
# card already at 94-99% utilisation the only way to reduce wall clock is to
# reduce work, and this is the one piece of it that is genuinely optional.
#
# `live` and `capacity` re-run from scratch and need only the seed, which is
# why they can go first at all. The baseline still runs, because the
# confirmatory grid wants it and the card would otherwise idle -- it is simply
# no longer between us and the answer.
say "=== 1/2 live:$AGENT + capacity, the roster-matched pair (the H3b test) ==="
if ! python -u -m collabengine.cli ablate --config "$CONFIG" \
     --modes live,capacity --agents "$AGENT" 2>&1 | tee -a "$LOG"; then
  say "FAILED ablation"
  exit 1
fi
say "H3B ARMS COMPLETE -- the contrast is readable now"

say "=== 2/2 baseline, $EPISODES episodes at seeds 1000-$((1000 + EPISODES - 1)) ==="
if ! python -u -m collabengine.cli pipeline --config "$CONFIG" \
     --phases baseline --episodes "$EPISODES" 2>&1 | tee -a "$LOG"; then
  say "FAILED baseline (the H3b arms above are unaffected)"
  exit 1
fi

say "ALL DONE -- python -m collabengine.cli analyze --config $CONFIG"
