#!/usr/bin/env bash
# Replaces queue_100.sh (WORKLOG (23), 2026-11-27). Priority order:
#   0  analyses of CelebA at 50 (the run queue_100 started) + pre-registered diversity test on all 50-seed data
#   A  dp-paper-nostyle (Maryam's dp-paper prompts on FP's weights, configs/<task>_nostyle.yaml) -> 50 seeds,
#      CIFAR-10 / pizza / teddy: is the dp-paper diversity effect the prompts or the style LoRA?
#   B  CIFAR-10 / pizza / teddy, fp / dp / dp-paper -> 100 seeds (paper's N)
#   C  dp-paper-nostyle -> 100 seeds
#   D  CelebA fp / dp / dp-paper -> 100 seeds
# Runs resume from records.jsonl (seed n -> same latent and prompt in every arm). After each phase: analyses,
# docs/auto_results.md regenerated, committed and pushed (only that file). Code/configs used by runs must not change
# while it runs. Progress: runs/main/status.log.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=$HOME/miniforge3/envs/dptig/bin/python
PIN=${PIN:-"taskset -c 16-31"}
M=runs/main
mkdir -p $M/analysis
log() { echo "$(date '+%F %T') $*" | tee -a $M/status.log; }
count() { cat "$1/records.jsonl" 2>/dev/null | wc -l; }
view() { case $1 in cifar10) echo cifar10;; imagenet_*) echo imagenet;; celebahq) echo celeba;; esac; }

run() {  # run <task> <arm> <N>; arm dp-paper-nostyle = method dp-paper with configs/<task>_nostyle.yaml
  local task=$1 arm=$2 n=$3 out=$M/${1}_${2} method=$2 cfg=configs/$1.yaml
  [ "$arm" = dp-paper-nostyle ] && { method=dp-paper; cfg=configs/${task}_nostyle.yaml; }
  [ "$(count $out)" -ge "$n" ] && return 0
  for i in 1 2 3; do
    log "$task $arm -> $n seeds, attempt $i (have $(count $out))"
    PYTHONFAULTHANDLER=1 $PIN $PY -m dptig.run --config $cfg --method $method --out $out --num-seeds $n \
      >> $M/${task}_${arm}.log 2>&1
    log "$task $arm exit $? (have $(count $out)/$n)"
    [ "$(count $out)" -ge "$n" ] && return 0
  done
  log "$task $arm FAILED to reach $n seeds"; return 1
}

analyze() {  # analyze <task> <arm>: FP vs arm, Eq. 6 + joint Table-3 (10 KMeans seeds), CPU only
  local task=$1 arm=$2 tag=$(date +%H%M)
  [ -s $M/${task}_fp/records.jsonl ] && [ -s $M/${task}_${arm}/records.jsonl ] || return 0
  $PIN $PY -m dptig.analyze --fp $M/${task}_fp --dp $M/${task}_${arm} --diversity --table3 --view $(view $task) \
    --kmeans-seeds 10 --device cpu --k 30 --m 20 \
    > $M/analysis/${task}_fp_vs_${arm}_n$(count $M/${task}_${arm})_$tag.json 2>> $M/analysis/errors.log \
    && log "analysis ${task} fp vs ${arm} written" || log "analysis ${task} fp vs ${arm} FAILED (see analysis/errors.log)"
}

extra() {  # extra <task> <N>: per-arm Table-3 clustering + pre-registered permutation test
  local task=$1 n=$2
  $PIN $PY scripts/t3_separate.py $task $(view $task) $n $M/analysis/${task}_t3_separate_n$n.json \
    >> $M/analysis/extra.log 2>> $M/analysis/errors.log && log "t3_separate $task n$n written" || log "t3_separate $task n$n FAILED"
  $PIN $PY scripts/div_perm.py $task $(view $task) $n $M/analysis/${task}_divperm_n$n.json \
    >> $M/analysis/extra.log 2>> $M/analysis/errors.log && log "div_perm $task n$n written" || log "div_perm $task n$n FAILED"
}

publish() {  # publish <phase>: regenerate docs/auto_results.md, commit only that file, push
  $PY scripts/auto_summary.py > docs/auto_results.md.tmp 2>> $M/analysis/errors.log \
    && mv docs/auto_results.md.tmp docs/auto_results.md || { log "auto_summary FAILED"; return 0; }
  git add docs/auto_results.md
  git commit -q -m "auto_results: queue_weekend phase $1 done" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" \
    -- docs/auto_results.md || { log "commit phase $1: nothing to commit"; return 0; }
  for i in 1 2 3; do
    git pull -q --rebase --autostash && git push -q && { log "phase $1 pushed"; return 0; }
    sleep 60
  done
  log "phase $1 push FAILED (committed locally)"
}

# never share the GPU with another queue / run
while pgrep -f "queue_100[.]sh|paper_queue[.]sh|overnight[.]sh" > /dev/null || pgrep -f "dptig[.]run " > /dev/null; do sleep 60; done
log "queue_weekend start"
T3="cifar10 imagenet_pizza imagenet_teddy"

# 0: CelebA at 50 (queue_100 stopped before analysing it) + pre-registered test on all 50-seed data
for a in fp dp dp-paper; do run celebahq $a 50; done
analyze celebahq dp; analyze celebahq dp-paper
for t in $T3 celebahq; do extra $t 50; done
publish 0

# A: no-style arm to 50
for t in $T3; do run $t dp-paper-nostyle 50; analyze $t dp-paper-nostyle; extra $t 50; done
publish A

# B: main arms to 100
for t in $T3; do
  for a in fp dp dp-paper; do run $t $a 100; done
  analyze $t dp; analyze $t dp-paper; extra $t 100
  publish "B ($t)"
done

# C: no-style arm to 100
for t in $T3; do run $t dp-paper-nostyle 100; analyze $t dp-paper-nostyle; extra $t 100; done
publish C

# D: CelebA to 100
for a in fp dp dp-paper; do run celebahq $a 100; done
analyze celebahq dp; analyze celebahq dp-paper; extra celebahq 100
publish D
log "QUEUE_WEEKEND DONE"
