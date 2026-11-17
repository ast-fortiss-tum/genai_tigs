#!/usr/bin/env bash
# Phase 1 grid: {logit,softmax}_margin x delta_scale {1,2}, CIFAR-10 FP-TIG, same seeds.
# Resumable: each config is re-launched (it resumes from records.jsonl) until it has N records, at most 3 attempts.
# Full output (incl. tracebacks) goes to runs/grid/<config>.log; exit codes to runs/grid/attempts.log.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/miniforge3/envs/dptig/bin/python}
N=${N:-5}
CONFIGS=${CONFIGS:-"logit_d1 logit_d2 softmax_d1 softmax_d2"}
# CPU pinning, e.g. PIN="taskset -c 16-31" (E-cores on the i9-13900KF box; see WORKLOG 2026-11-17 (4))
PIN=${PIN:-}
mkdir -p runs/grid
for c in $CONFIGS; do
  out=runs/grid/cifar10_fp_$c
  for attempt in 1 2 3; do
    have=$(cat "$out/records.jsonl" 2>/dev/null | wc -l)
    [ "$have" -ge "$N" ] && break
    echo "$(date +%T) $c attempt $attempt start (have $have/$N)" >> runs/grid/attempts.log
    PYTHONFAULTHANDLER=1 $PIN $PY -m dptig.run --config configs/grid/cifar10_$c.yaml --method fp --out "$out" --num-seeds "$N" \
      >> runs/grid/cifar10_fp_$c.log 2>&1
    rc=$?
    echo "$(date +%T) $c attempt $attempt exit $rc pin='$PIN'" >> runs/grid/attempts.log
  done
done
