#!/usr/bin/env bash
# Overnight queue (2026-11-17 -> 18), settings from the Phase 1 grid: delta_scale 2.0, logit_margin (configs/).
# Runs grow in balanced rounds so that whenever it stops, FP and DP (and dp-template) are paired on the same latents.
# Every run resumes from its records.jsonl; a failed run is retried (3x), then logged and skipped, never fatal.
# Logs: runs/main/<task>_<method>.log, progress: runs/main/status.log, analyses: runs/main/analysis/*.json
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniforge3/envs/dptig/bin/python
PIN=${PIN:-"taskset -c 16-31"}
M=runs/main
mkdir -p $M/analysis
log() { echo "$(date '+%F %T') $*" | tee -a $M/status.log; }
count() { cat "$1/records.jsonl" 2>/dev/null | wc -l; }

run() {  # run <task> <method> <N>
  local task=$1 method=$2 n=$3 out=$M/${1}_${2}
  [ "$(count $out)" -ge "$n" ] && return 0
  for i in 1 2 3; do
    log "$task $method -> $n seeds, attempt $i (have $(count $out))"
    PYTHONFAULTHANDLER=1 $PIN $PY -m dptig.run --config configs/$task.yaml --method $method --out $out --num-seeds $n \
      >> $M/${task}_${method}.log 2>&1
    local rc=$?
    log "$task $method exit $rc (have $(count $out)/$n)"
    [ "$(count $out)" -ge "$n" ] && return 0
  done
  log "$task $method FAILED to reach $n seeds"
  return 1
}

analyze() {  # analyze <task> <arm>  (FP vs arm), CPU so it never competes with the GPU run
  local task=$1 arm=$2 tag=$(date +%H%M)
  [ -s $M/${task}_fp/records.jsonl ] && [ -s $M/${task}_${arm}/records.jsonl ] || return 0
  $PIN $PY -m dptig.analyze --fp $M/${task}_fp --dp $M/${task}_${arm} --diversity --device cpu --k 30 --m 20 \
    > $M/analysis/${task}_fp_vs_${arm}_n$(count $M/${task}_fp)_$tag.json 2>> $M/analysis/errors.log \
    && log "analysis ${task} fp vs ${arm} written" || log "analysis ${task} fp vs ${arm} FAILED (see analysis/errors.log)"
}

seedcheck() {
  local task=$1
  [ -f runs/seedcheck_$task/report.json ] && return 0
  log "seed check $task"
  $PIN $PY -m dptig.seed_check --config configs/$task.yaml --per-class 20 --out runs/seedcheck_$task >> $M/seedcheck_$task.log 2>&1
  log "seed check $task: $(python3 -c "import json;print(round(json.load(open('runs/seedcheck_$task/report.json'))['overall_rate'],3))" 2>/dev/null || echo FAILED)"
}

# 0 wait for the CelebA pipeline and for any run left over from a previous queue instance (they own the GPU)
while pgrep -f "celeba_replicate[.]sh" > /dev/null || pgrep -f "dptig[.]run " > /dev/null; do sleep 30; done
log "start (CelebA pipeline finished)"

# reuse CIFAR-10 FP from the grid: logit_d2 has exactly these settings (configs diff only in comments)
if [ ! -d $M/cifar10_fp ] && [ -s runs/grid/cifar10_fp_logit_d2/records.jsonl ]; then
  cp -r runs/grid/cifar10_fp_logit_d2 $M/cifar10_fp && log "cifar10 fp: reused grid logit_d2 ($(count $M/cifar10_fp) seeds)"
fi

# dp-template is pointless when its prompts equal the DP prompts for every latent (pizza: 100/100; WORKLOG (10))
same_prompts() {  # same_prompts <task> <N> -> exit 0 if DP and dp-template prompts are identical for latents < N
  $PY - "$1" "$2" <<'PYEOF' 2>/dev/null
import sys
from pathlib import Path
from dptig.config import load_config
from dptig.prompts import make_provider
from dptig.llm_prompt import build_prompt_generator
t, n = sys.argv[1], int(sys.argv[2])
c = load_config(f"configs/{t}.yaml")
dp = make_provider("dp", c, build_prompt_generator(c.llm, Path(f"prompts/{c.name}_dp.jsonl"), None))
tp = make_provider("dp-template", c, None)
sys.exit(0 if all(dp.get(i).prompt.strip().lower() == tp.get(i).prompt.strip().lower() for i in range(n)) else 1)
PYEOF
}
template() {  # template <task> <N>: run dp-template unless it would duplicate DP
  if same_prompts "$1" "$2"; then log "$1 dp-template skipped: prompts identical to DP for all $2 latents"; return 0; fi
  run "$1" dp-template "$2" && analyze "$1" dp-template
}

celeba_ok() { python3 -c "import json,sys;sys.exit(0 if json.load(open('runs/seedcheck_celebahq/report.json'))['overall_rate']>=0.5 else 1)" 2>/dev/null; }

# Round 1 (5 seeds): every task x every arm; CelebA FP only (no DP descriptors yet) and only if its seeds are valid
run cifar10 dp 5; analyze cifar10 dp
template cifar10 5
for t in imagenet_pizza imagenet_teddy; do
  seedcheck $t
  run $t fp 5; run $t dp 5; analyze $t dp
  template $t 5
done
if celeba_ok; then run celebahq fp 5; else log "celebahq: seed validity < 0.5 or no seed check, FP skipped"; fi
log "round 1 done"

# Rounds 2+: FP and DP grow together (main comparison), dp-template follows at the first round only
for n in 15 30 50 100; do
  for t in cifar10 imagenet_pizza imagenet_teddy; do
    run $t fp $n; run $t dp $n; analyze $t dp
    [ "$n" -le 15 ] && template $t $n
  done
  if celeba_ok; then run celebahq fp $n; fi
  log "round n=$n done"
done
log "ALL DONE"
