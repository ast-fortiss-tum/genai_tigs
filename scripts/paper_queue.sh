#!/usr/bin/env bash
# Queue after Maryam's code (2026-11-24, WORKLOG (17)): dp-paper arm (her prompts + style LoRA) on all tasks, teddy FP
# rerun with the correct LoRA trigger, CelebA FP/DP/dp-paper with her LoRA. Rounds of 15 then 30 seeds so arms stay
# paired whenever it stops. Resumes from records.jsonl; failed runs retried 3x, never fatal.
# Progress: runs/main/status.log; analyses (incl. Table-3 cluster diversity): runs/main/analysis/*.json
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniforge3/envs/dptig/bin/python
PIN=${PIN:-"taskset -c 16-31"}
M=runs/main
mkdir -p $M/analysis
log() { echo "$(date '+%F %T') $*" | tee -a $M/status.log; }
count() { cat "$1/records.jsonl" 2>/dev/null | wc -l; }
view() { case $1 in cifar10) echo cifar10;; imagenet_*) echo imagenet;; celebahq) echo celeba;; esac; }

run() {  # run <task> <method> <N>
  local task=$1 method=$2 n=$3 out=$M/${1}_${2}
  [ "$(count $out)" -ge "$n" ] && return 0
  for i in 1 2 3; do
    log "$task $method -> $n seeds, attempt $i (have $(count $out))"
    PYTHONFAULTHANDLER=1 $PIN $PY -m dptig.run --config configs/$task.yaml --method $method --out $out --num-seeds $n \
      >> $M/${task}_${method}.log 2>&1
    log "$task $method exit $? (have $(count $out)/$n)"
    [ "$(count $out)" -ge "$n" ] && return 0
  done
  log "$task $method FAILED to reach $n seeds"; return 1
}

analyze() {  # analyze <task> <arm>: FP vs arm, Eq. 6 + Table-3 cluster diversity (10 KMeans seeds), CPU only
  local task=$1 arm=$2 tag=$(date +%H%M)
  [ -s $M/${task}_fp/records.jsonl ] && [ -s $M/${task}_${arm}/records.jsonl ] || return 0
  $PIN $PY -m dptig.analyze --fp $M/${task}_fp --dp $M/${task}_${arm} --diversity --table3 --view $(view $task) \
    --kmeans-seeds 10 --device cpu --k 30 --m 20 \
    > $M/analysis/${task}_fp_vs_${arm}_n$(count $M/${task}_${arm})_$tag.json 2>> $M/analysis/errors.log \
    && log "analysis ${task} fp vs ${arm} written" || log "analysis ${task} fp vs ${arm} FAILED (see analysis/errors.log)"
}

while pgrep -f "dptig[.]run " > /dev/null; do sleep 30; done
# archive runs made with superseded settings (kept, not deleted)
[ -d $M/imagenet_teddy_fp ] && [ ! -d $M/old_imagenet_teddy_fp_wrongtrigger ] && \
  mv $M/imagenet_teddy_fp $M/old_imagenet_teddy_fp_wrongtrigger && log "archived teddy FP (trigger 1teddy_toy)"
[ -d $M/celebahq_fp ] && [ ! -d $M/old_celebahq_fp_ourlora ] && \
  mv $M/celebahq_fp $M/old_celebahq_fp_ourlora && log "archived CelebA FP (our LoRA)"
log "paper queue start"

for n in 15 30; do
  run imagenet_teddy fp $n; analyze imagenet_teddy dp
  for t in imagenet_teddy imagenet_pizza cifar10; do run $t dp-paper $n; analyze $t dp-paper; done
  for a in fp dp dp-paper; do run celebahq $a $n; done
  analyze celebahq dp; analyze celebahq dp-paper
  log "paper round n=$n done"
done
log "PAPER QUEUE DONE"
