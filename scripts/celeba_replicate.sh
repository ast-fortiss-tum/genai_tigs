#!/usr/bin/env bash
# CelebA-HQ replication pipeline (Phase 3 without the authors' weights). Idempotent: each step is skipped when its
# output exists, so the script can simply be re-run after a crash. GPU steps wait until the Phase 1 grid is done.
#   1 pack CelebA (CPU)  2 wait for grid  3 train classifier  4 select LoRA images (classifier filter)
#   5 train LoRA (kohya)  6 seed-validity check
# Logs: runs/celeba/*.log, progress: runs/celeba/status.log. Pinned to E-cores (see WORKLOG 2026-11-17 (4)).
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PY=$HOME/miniforge3/envs/dptig/bin/python
PIN=${PIN:-"taskset -c 16-31"}
D=$HOME/data
L=runs/celeba
mkdir -p $L
log() { echo "$(date '+%F %T') $*" | tee -a $L/status.log; }
retry() {  # retry N name cmd...
  local n=$1 name=$2; shift 2
  for i in $(seq 1 "$n"); do
    log "$name attempt $i start"
    PYTHONFAULTHANDLER=1 $PIN "$@" >> $L/$name.log 2>&1
    local rc=$?
    log "$name attempt $i exit $rc"
    [ $rc -eq 0 ] && return 0
  done
  return 1
}

# 1 pack CelebA
if [ ! -f $D/celeba_packed/test.npz ]; then
  retry 1 prepare $PY -m dptig.celeba_classifier prepare --src $D/celeba_flwr --dst $D/celeba_packed || { log "prepare FAILED"; exit 1; }
fi

# 2 wait for the grid (GPU is full while it runs)
while pgrep -f "runs_grid[.]sh" > /dev/null; do sleep 60; done
log "grid finished, GPU free"

# 3 classifier (resumes from last.pt)
if [ ! -f weights/classifiers/celeba/metrics.json ]; then
  retry 3 classifier $PY -m dptig.celeba_classifier train --celeba $D/celeba_packed \
    --hq $D/celebamask_hq/CelebAMask-HQ --out weights/classifiers/celeba --workers 12 || { log "classifier FAILED"; exit 1; }
fi
log "classifier: $(python3 -c "import json;d=json.load(open('weights/classifiers/celeba/metrics.json'));print('test',round(d['test']['mean_acc'],4),'hq',round(d['hq_test']['mean_acc'],4),'targets_hq',{k:round(v,3) for k,v in d['hq_test']['targets'].items()})")"

# 4 LoRA training images
if [ ! -f $D/celeba_lora_train/selection.json ]; then
  rm -rf $D/celeba_lora_train
  retry 1 lora_data $PY -m dptig.celeba_lora_data --hq $D/celebamask_hq/CelebAMask-HQ --out $D/celeba_lora_train \
    --clf weights/classifiers/celeba/resnet50_celeba40.pth || { log "lora_data FAILED"; exit 1; }
fi

# 5 LoRA (restart from scratch on failure; ~1 h)
if [ ! -f weights/lora/celebahq_lorav1.5-000005.safetensors ]; then
  retry 2 lora_train bash scripts/train_celeba_lora.sh || { log "lora_train FAILED"; exit 1; }
fi

# 6 seed validity (paper Table 2: 93-94% of CelebA-HQ seeds valid)
if [ ! -f runs/seedcheck_celebahq/report.json ]; then
  retry 2 seedcheck $PY -m dptig.seed_check --config configs/celebahq.yaml --per-class 20 --out runs/seedcheck_celebahq \
    || { log "seedcheck FAILED"; exit 1; }
fi
log "seed check: $(python3 -c "import json;d=json.load(open('runs/seedcheck_celebahq/report.json'));print({k:(v['valid'],v['n']) for k,v in d.items() if isinstance(v,dict)}, 'overall', round(d['overall_rate'],3))")"
log "DONE"
