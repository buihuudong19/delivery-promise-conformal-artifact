#!/usr/bin/env bash
# Full reproduction on Linux or macOS. Extra arguments are passed to run_pipeline.py, e.g. ./run_pipeline.sh --quick
# Without arguments the results of the previous run (outputs/, outputs_extra/) are moved to _archive/run_<date>_<time>
# (the 2 most recent archives are kept), so a full run starts clean;
# ./run_pipeline.sh --resume continues an interrupted full run without archiving (fitted origins are kept).
set -euo pipefail
cd "$(dirname "$0")"
RESUME=0; ARGS=()
for a in "$@"; do
  if [ "$a" = "--resume" ]; then RESUME=1; else ARGS+=("$a"); fi
done
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip --quiet
.venv/bin/python -m pip install -r requirements-lock.txt --quiet || .venv/bin/python -m pip install -r requirements.txt --quiet
if [ ${#ARGS[@]} -eq 0 ] && [ "$RESUME" -eq 0 ]; then
  echo "Full run: clearing the results of the previous run (kept in _archive, last 2 runs) ..."
  .venv/bin/python tools/clear_previous_run.py
fi
mkdir -p outputs/logs
TEE=(tee); { [ "$RESUME" -eq 1 ] || [ "$#" -gt 0 ]; } && TEE=(tee -a)   # a run with arguments appends to the log
.venv/bin/python -u run_pipeline.py ${ARGS[@]+"${ARGS[@]}"} 2>&1 | "${TEE[@]}" outputs/logs/pipeline.log
