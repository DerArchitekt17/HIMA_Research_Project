#!/usr/bin/env bash
set -u  # don't use -e; we want to continue even if wandb sync fails sometimes

# Usage:
#   Loop mode (background):  nohup ./wandb_sync.sh > wandb_sync.log 2>&1 &
#   One-shot mode (cron):    ./wandb_sync.sh --once
#
# Cron setup (every 5 minutes):
#   crontab -e
#   */5 * * * * cd /work/hanken/HIMA_Research_Project && ./wandb_sync.sh --once >> wandb_sync.log 2>&1

ENV_NAME="hima_research"
INTERVAL=30

# All experiment directories to sync (Outcomment for more scoping)
EXPERIMENTS="single_agent multi_agents swarm_agents single_agent_small multi_agents_small swarm_agents_small"
#EXPERIMENTS="single_agent multi_agents swarm_agents"
#EXPERIMENTS="single_agent_small multi_agents_small swarm_agents_small"

# Stop only after this many consecutive empty checks (loop mode only).
# Example: 10 checks * 30s = 5 minutes with nothing to sync -> exit
EMPTY_LIMIT=10

# Parse --once flag
ONCE=false
if [[ "${1:-}" == "--once" ]]; then
  ONCE=true
fi

# Environment activation: try conda first, then fall back to PATH
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  echo "Starting W&B sync loop in env: $ENV_NAME"
elif ! command -v wandb &>/dev/null; then
  echo "ERROR: wandb not found. Activate a virtualenv or install wandb first." >&2
  exit 1
fi

# Resolve project root (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

sync_all () {
  local synced=0
  for exp in $EXPERIMENTS; do
    shopt -s nullglob
    local runs=( ${exp}/wandb/offline-run-* )
    shopt -u nullglob
    if [[ ${#runs[@]} -gt 0 ]]; then
      echo "[$(date)] [${exp}] Found ${#runs[@]} offline run(s). Syncing..."
      wandb sync --include-offline ${exp}/wandb/offline-run-* \
        || echo "[$(date)] [${exp}] wandb sync returned non-zero; continuing."
      synced=$((synced + ${#runs[@]}))
    fi
  done
  echo "$synced"
}

# One-shot mode: sync once and exit (for cron)
if [[ "$ONCE" == true ]]; then
  sync_all > /dev/null
  exit 0
fi

# Loop mode: continuous sync with auto-exit on idle
echo "Experiments: $EXPERIMENTS"
echo "Interval: ${INTERVAL}s"
echo "Empty limit: $EMPTY_LIMIT cycles"
echo

empty=0

while true; do
  synced="$(sync_all)"

  if [[ "$synced" -eq 0 ]]; then
    ((empty++))
    echo "[$(date)] Nothing to sync. empty=$empty/$EMPTY_LIMIT"
    if [[ "$empty" -ge "$EMPTY_LIMIT" ]]; then
      echo "[$(date)] No offline runs for $((EMPTY_LIMIT*INTERVAL)) seconds. Exiting."
      exit 0
    fi
  else
    empty=0
  fi

  sleep "$INTERVAL"
done
