#!/usr/bin/env bash
# Extend all paper arms from 30 to the paper's 100 latents (paper Sec. 5: "For each task, we generated 100 initial
# latent vectors"; Maryam's scripts: imgs_to_samp = 100). WORKLOG (22). Runs resume from records.jsonl (seed n ->
# latent base_seed+n, prompt n from prompts/*.jsonl), so seeds 0-29 are kept and 30-99 are appended; arms stay paired.
# Rounds of 50 then 100 so every arm is paired whenever it stops; fast tasks first, CelebA (~20 min/seed/arm) last.
# Code/configs must not change while it runs. Progress: runs/main/status.log; analyses: runs/main/analysis/*.json
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

# never share the GPU with another queue
while pgrep -f "paper_queue[.]sh|overnight[.]sh" > /dev/null || pgrep -f "dptig[.]run " > /dev/null; do sleep 60; done
log "queue_100 start"

for n in 50 100; do
  for t in cifar10 imagenet_pizza imagenet_teddy celebahq; do
    for a in fp dp dp-paper; do run $t $a $n; done
    analyze $t dp; analyze $t dp-paper
  done
  log "queue_100 round n=$n done"
done
log "QUEUE_100 DONE"
