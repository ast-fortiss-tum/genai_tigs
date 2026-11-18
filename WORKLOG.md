# Work log: reimplementing DP-TIG ("Prompt Diversity for Diffusion-Based Test Input Generation")

Newest entries first. Every piece of work adds an entry here: what was done, decisions and deviations from the
paper/legacy code, what was verified vs. not, and what is open.

**Goal.** Reimplement the paper's extension (DP-TIG: LLM-guided target-descriptor prompts + the same SD/GA search,
evaluated against FP-TIG on CIFAR-10, ImageNet pizza/teddy and CelebA-HQ) by reusing this repo's code. The authors'
extension code/models are not available.

**Status at a glance** (updated 2026-11-18)

| Phase | Scope | State |
|---|---|---|
| 0 | Refactor legacy SD TIG into reusable `dptig/` package (FP-TIG), configs, tests | done; runs on CUDA (RTX 4090) |
| 1 | Reproduce FP-TIG numbers (paper Table 2, FP rows) | **done for CIFAR-10** (30 seeds match Table 2); settings from a 2x2 grid: `delta_scale 2.0`, `logit_margin` (= legacy code) |
| 2 | DP-TIG: LLM prompts, descriptors, providers, prompt cache | done; 100 cached prompts per task (CelebA: no descriptors yet) |
| 2b | LLM-free `dp-template` ablation arm | done; skipped automatically where identical to DP (pizza) |
| 3 | CelebA-HQ (classifier, LoRA, logit fitness) | **replicated** (authors' weights unavailable): classifier 91.3% on CelebA-HQ, LoRA seed validity 60/60; FP runs 15 seeds |
| 4 | Diversity metrics (CLIP/ResNet-50), statistics, MTurk assets | metrics/stats/analysis done and run on real images; MTurk + label preservation todo (need human data) |
| 5 | Full experiment (100 latents x {FP,DP} x {low,high} x 4 tasks) | **running** in rounds (15 seeds everywhere, CIFAR-10 at 30); `scripts/overnight.sh`, `runs/main/` |

## QUESTIONS FOR THE 2026-11-18 MEETING (Andrea + V. Riccio)
Evidence: entries (6)-(11) below and `python -m dptig.summarize runs/main`. Numbers are from our reimplementation
(15 seeds per task/arm, CIFAR-10 FP at 30), preliminary; no FP-vs-DP difference is significant yet.

**Paper / method (co-authors)**
1. **Diversity scale vs Table 3.** Our Eq. 6 (mean of 1 - cosine similarity over all distinct image pairs, checked
   against a naive loop) gives D ~0.15-0.30 (CLIP and ResNet-50, all tasks). Table 3 reports ~0.8-0.95. D = 0.95 would
   mean same-class images are nearly orthogonal in CLIP space, which is implausible. *How was Table 3 computed?*
   (Different normalisation, features, pairs, or a scaled score?) Blocks any comparison with Table 3.
   Related: what is one diversity "score" for the paired t-test? We assume 30 random subsets of 20 images (entry 2026-11-15 (2)).
2. **Iteration-1 "failures" from the guidance switch.** Seeds use guidance 3.5, mutants 1.4 (Sec. 4.1.1). In 25-35%
   of valid seeds, in every arm, the very first mutant generation is already misclassified: the label flips because
   of the guidance change, not because of the search (e.g. CIFAR-10 FP at 30 seeds: 6-7 of 28). This lowers mean
   iterations for all arms. *Exclude, report separately, or generate seeds at the mutation guidance?*
3. **The LLM mostly reproduces the template.** DP prompt == `dp-template` prompt (case-insensitive) for 100/100 pizza,
   58/100 CIFAR-10, 28/100 teddy latents (only 3 / 40 / 9 distinct DP texts). Where identical, results are identical
   (pipeline is deterministic). So far `dp-template` ~ DP on every metric. This supports the remark that the LLM is a
   combinatorial step. *Keep "LLM-guided" framing, make prompts more varied (temperature, more descriptors), or
   reframe the contribution around descriptors rather than the LLM?*
4. **Preliminary direction of the results contradicts the paper.** At 15 seeds DP's D is lower than FP's in 9/12
   comparisons (paper: DP clearly more diverse), DP loses more seeds on CIFAR-10 (11/15 vs 14/15) and teddy (8/15 vs
   11/15; paper: DP more valid seeds), no iteration difference (Mann-Whitney p >= 0.34). The CIFAR-10 test at 30 seeds
   (first one with enough images for the diversity t-test) is expected ~14:00 today. *How do we proceed if it holds?*
5. **Algorithm 1 / Eq. 3 vs the code that produced Table 2.** Our 2x2 grid (entry (6)) says Table 2 matches the
   legacy code (delta = 2x the extent, raw-logit margin), not the paper text (1x, softmax margin). FP at 30 seeds with
   legacy settings matches Table 2 (low: 89% fail / 78 it vs 89.66% / 85.63). *Fix the text, or rerun with the
   paper's settings?* Ask Maryam to confirm.

**Data / setup (Andrea)**
6. **CelebA-HQ LoRA bias.** Our retrained LoRA's "eye-glasses" seeds are almost all men with sunglasses (many
   dark-skinned or black-and-white photo style); "curtain-bangs" seeds are mostly blond women (CelebA skew of the 38
   training images per concept). Affects diversity and what failures look like. *Retrain on images balanced by
   gender and eyeglasses-vs-sunglasses (~1 h GPU)?* Also still ask Maryam for the original CelebA weights.
7. **CelebA DP descriptors** are not given in the paper ("attribute-specific constraints"). Proposal: derive candidates
   with `dptig/descriptors.py` (BLIP captions of generated seeds), co-authors pick 3 per attribute. Until then CelebA
   runs FP only.
8. **Priorities for the GPU.** Queue now: CIFAR-10 DP to 30 (~14:00), then pizza/teddy FP+DP to 30, CelebA FP to 30
   (~10 h more), then 50 and 100. Full 100-seed experiment ~8 GPU-days on this box. *Same order, or change (e.g.
   settle questions 1-3 first, as they may change what we run)?*
9. **Machine.** i9-13900KF with BIOS F6 (2023): random crashes on P-cores; runs are pinned to E-cores (no crash in
   ~22 h). Recommend a BIOS update + CPU stress test before the long run.

**Not answerable by us:** MTurk validity / label preservation (RQ4-5) need a new human study.

## REPRODUCTION STATUS BY RQ (2026-11-18, ~11:15; CIFAR-10 / pizza / teddy 30-seed results added 13:00 / 15:20 / 16:30)
Progress: ~15% of Phase 5 (~120 of 800 FP/DP seed-runs = 100 latents x {FP,DP} x 4 tasks); 15 seeds per task/arm
(CIFAR-10 FP at 30), CelebA FP only. Settings: delta_scale 2.0, logit_margin (entry (6)). Numbers: entry (11) and
`python -m dptig.summarize runs/main`.

| RQ | Paper's claim | Our status | Our result so far (15 seeds) |
|---|---|---|---|
| **RQ1** Seed generation (% valid seeds) | DP produces valid seeds as well or better | Measured on 3 tasks; CelebA FP only | *30 seeds: CIFAR-10 DP 83% vs FP 93% (p=0.42; paper 84 vs 87, same direction); pizza DP 60% vs FP 77% (p=0.27; paper 85 vs 73, opposite); **teddy DP 50% vs FP 83%, p=0.013, significant, opposite to the paper (92 vs 87)**.* At 15 seeds: DP loses more seeds. CIFAR-10 11/15 vs 14/15, teddy 8/15 vs 11/15, pizza 10/15 vs 11/15 (paper: DP higher on pizza 85 vs 73, teddy 92 vs 87). Not significant (p >= 0.33) |
| **RQ2** Effectiveness (% failures found) | DP matches or beats FP | Measured on 3 tasks | **Agrees**: both ~100% everywhere, no difference. FP matches Table 2 (CIFAR-10 FP at 30 seeds: low 89% vs 89.66%) |
| **RQ3** Efficiency (iterations to first failure) | DP generally fewer | Measured on 3 tasks | *CIFAR-10 at 30 seeds: **same direction as the paper**, DP 68.5 vs FP 77.9 (low), 13.5 vs 16.5 (high), not significant (p=0.50/0.56). Pizza at 30: DP 56.9 vs FP 66.1 (p=0.83), paper DP much slower (123 vs 84), so opposite. Teddy at 30: DP 45.2 vs FP 38.2 (p=0.53), paper DP faster.* At 15 seeds, inconclusive and mostly opposite: low level CIFAR-10 DP 91 vs FP 79, teddy 60 vs 46, pizza 53 vs 63 (paper: DP faster on CIFAR-10/teddy, slower on pizza; opposite on all 3). Not significant (p >= 0.34). 25-35% iteration-1 failures distort all arms (question 2) |
| **RQ4** Validity (human, MTurk) | DP more valid inputs | **Not started** | Needs a new MTurk study (legacy survey HTML exists, not adapted) |
| **RQ5** Label preservation (human) | Mixed | **Not started** | Same MTurk study |
| **RQ6** Image diversity (CLIP / ResNet-50) | DP more diverse | First test done (CIFAR-10, 30 seeds) | **Task-dependent**: pizza at 30 seeds DP more diverse in 4/4 (descriptive, paper's way); teddy about even (2/4). CIFAR-10 at 30 seeds, DP never more diverse, significantly less in 2 of 4 (CLIP low d=-0.68, ResNet high d=-1.6), but p-values are inflated by overlapping subsamples (entry (12)). At 15 seeds DP's D lower in 9 of 12. Scale differs from Table 3 (~0.25 vs ~0.9, question 1). CLIP and ResNet D uncorrelated (r=0.02) |

- **Solid:** the FP baseline. Our reimplementation reproduces Table 2's FP rows on CIFAR-10, pizza and teddy.
- **Open:** every DP-vs-FP claim is so far directionally against the paper but not significant; 15 seeds is too few.
  CIFAR-10 at 30 seeds is the first real test; RQ6 cannot be compared with Table 3 until question 1 is answered.
- **Missing:** CelebA DP (descriptors, question 7), the human study (RQ4/RQ5), ~85% of the seed-runs
  (~6-7 GPU-days at the current rate, less if priorities change).

## RESUME HERE (2026-11-18, GPU machine)
State: `main` has local commits (not pushed). Env: conda env `dptig` (`~/miniforge3`); kohya env `kohya` for LoRA
training. 51 tests pass (`python -m unittest discover -s tests`). Long runs: pin to E-cores (`taskset -c 16-31`).
- Running: `scripts/overnight.sh` (rounds of FP/DP to 15/30/50/100 seeds), progress in `runs/main/status.log`,
  analyses in `runs/main/analysis/`, summary table: `python -m dptig.summarize runs/main`.
- Next after the meeting: act on the questions above (diversity definition, iteration-1 handling, prompt variety,
  CelebA LoRA rebalancing + descriptors), then continue Phase 5. MTurk assets remain (Phase 4).

---

## 2026-11-18 (14): ImageNet teddy FP vs DP at 30 seeds -- first significant difference (RQ1, against the paper)
`runs/main/analysis/imagenet_teddy_fp_vs_dp_n30_1628.json`:

| | DP | FP | test | paper DP vs FP |
|---|---|---|---|---|
| seeds valid (RQ1) | **15/30 (50%)** | **25/30 (83%)** | **Fisher p=0.013** | 92 vs 87 |
| failures low / high (RQ2) | 15/15, 15/15 | 25/25, 25/25 | p=1.0 | 98.91/100 vs 98.85/100 |
| mean (median) iters low (RQ3) | 45.2 (39) | 38.2 (23) | MW p=0.53 | 40.79 vs 48.45 |
| mean (median) iters high | 4.0 (3) | 4.6 (2) | MW p=0.98 | 4.88 vs 6.63 |
| CLIP D low / high (RQ6) | 0.273 / 0.274 | 0.271 / 0.264 | no test (15 DP images) | DP higher |
| ResNet D low / high | 0.279 / 0.297 | 0.294 / 0.295 | no test | DP higher |

- **RQ1, significant, opposite to the paper:** DP teddy prompts ("on a bed", "sitting on a chair", ...) lose half
  the seeds (classifier predicts e.g. class 191/260 instead of teddy). Fewer valid seeds also means fewer failure
  images for RQ6 (only 15).
- RQ3: no difference. RQ6: about even (DP higher in 2/4, descriptive).
- Teddy FP reproduces Table 2's FP row (83% seeds vs 87; 100% failures; high iters 4.6 vs 6.63). Iteration-1 failures
  are frequent for teddy FP (7 low / 10 high of 25).
- Queue: CelebA FP to 30 started 16:29 (~4.5 h), end of the 30-seed round ~21:00.

## 2026-11-18 (13): ImageNet pizza FP vs DP at 30 seeds
`runs/main/analysis/imagenet_pizza_fp_vs_dp_n30_1520.json`:

| | DP | FP | test | paper DP vs FP |
|---|---|---|---|---|
| seeds valid (RQ1) | 18/30 (60%) | 23/30 (77%) | Fisher p=0.27 | 85 vs 73 |
| failures low / high (RQ2) | 18/18, 18/18 | 23/23, 23/23 | p=1.0 | 88.24/100 vs 97.26/100 |
| mean (median) iters low (RQ3) | 56.9 (50) | 66.1 (51) | MW p=0.83 | 123.28 vs 83.60 |
| mean (median) iters high | 9.6 (8) | 10.9 (7) | MW p=0.87 | 20.14 vs 12.53 |
| CLIP D low / high (RQ6) | 0.167 / 0.193 | 0.151 / 0.182 | no test (18 DP images <= m) | DP higher |
| ResNet D low / high | 0.201 / 0.237 | 0.182 / 0.230 | no test | DP higher |

- Pizza FP reproduces Table 2's FP row (77% seeds vs 73; 100% failures; high iters 10.9 vs 12.53).
- RQ6: pizza goes **the paper's way** (DP more diverse in 4/4, descriptive), unlike CIFAR-10 (entry (12)) -> the
  diversity effect is task-dependent so far. Pizza DP uses only 3 distinct prompts (the descriptors themselves).
- RQ1/RQ3: pizza **contradicts** the paper: DP loses more seeds and is not slower (paper: DP 85% seeds, 123 iters).
- Next: teddy FP/DP to 30 (started 15:21), then CelebA FP to 30.

## 2026-11-18 (12): CIFAR-10 FP vs DP at 30 seeds (first diversity test with enough images)
`runs/main/analysis/cifar10_fp_vs_dp_n30_1300.json`:

| | DP | FP | test | paper DP vs FP |
|---|---|---|---|---|
| seeds valid (RQ1) | 25/30 (83%) | 28/30 (93%) | Fisher p=0.42 | 84 vs 87 |
| low: failures (RQ2) | 22/25 | 25/28 | p=1.0 | 92.86 vs 89.66 |
| low: mean (median) iters (RQ3) | 68.5 (38) | 77.9 (64) | MW p=0.50, d=-0.12 | 74.73 vs 85.63 |
| high: mean (median) iters (RQ3) | 13.5 (9) | 16.5 (10) | MW p=0.56, d=-0.16 | 9.89 vs 14.18 |
| CLIP D low (RQ6) | 0.236 | 0.243 | paired t p=0.0008, d=-0.68 | DP higher |
| ResNet D low | 0.329 | 0.325 | p=0.08 | DP higher |
| CLIP D high | 0.250 | 0.249 | p=0.61 | DP higher |
| ResNet D high | 0.307 | 0.328 | p<0.001, d=-1.6 | DP higher |

- RQ1/RQ3 now go **in the paper's direction** (DP ~10% fewer iterations, almost the same gap as Table 2), not
  significant. At 15 seeds RQ3 pointed the other way -> 15-seed directions were noise.
- RQ6 **contradicts the paper**: DP never more diverse; significantly less in 2/4 comparisons.
- **Caveat (important):** diversity p-values come from k=30 subsets of m=20 images drawn from only 22-28 images per
  arm. The subsets overlap almost entirely, so they are not independent and the paired t-test overstates
  significance. Direction is reliable, the p-values are not. Needs the co-authors' definition of one diversity score
  (question 1); alternatives: one D per run repeat, bootstrap CI of the difference, or per-class D.
- CLIP vs ResNet diversity correlation r=0.02 (Pearson) / 0.03 (Spearman) at both levels: the two measures do not
  agree here (the paper reports their correlation as consistency evidence).
- Iteration-1 failures: DP 8 (low) / 9 (high) of 25, FP 6 / 7 of 28.

## 2026-11-18 (11): Overnight results, 15 seeds per task x arm (delta_scale 2.0, logit_margin)
Queue ran 17:56 -> 08:09 with **no crash** (E-core pinning, ~19 h crash-free in total). 30-seed round started 08:09.
`python -m dptig.summarize runs/main` (new) gives:

| task | arm | seeds valid | low: fail | low: mean (median) it | low: it=1 | high: fail | high: mean (median) it | high: it=1 | paper %seeds / low fail%, it / high fail%, it |
|---|---|---|---|---|---|---|---|---|---|
| celebahq | fp | 15/15 | 9/15 | 121.1 (116) | 1 | 15/15 | 26.9 (23) | 1 | 94 / 24.73, 18.22 / 91.47, 56.08 |
| cifar10 | fp | 14/15 | 13/14 | 78.8 (67) | 3 | 14/14 | 11.1 (9) | 4 | 87 / 89.66, 85.63 / 100, 14.18 |
| cifar10 | dp | 11/15 | 11/11 | 91.1 (71) | 4 | 11/11 | 12.4 (9) | 4 | 84 / 92.86, 74.73 / 100, 9.89 |
| cifar10 | dp-template | 11/15 | 10/11 | 78.1 (68) | 4 | 11/11 | 12.2 (9) | 4 |  |
| imagenet_pizza | fp | 11/15 | 11/11 | 63.1 (25) | 3 | 11/11 | 11.1 (4) | 3 | 73 / 97.26, 83.6 / 100, 12.53 |
| imagenet_pizza | dp | 10/15 | 10/10 | 52.7 (37) | 3 | 10/10 | 10.0 (7) | 4 | 85 / 88.24, 123.28 / 100, 20.14 |
| imagenet_teddy | fp | 11/15 | 11/11 | 45.7 (28) | 3 | 11/11 | 5.0 (4) | 4 | 87 / 98.85, 48.45 / 100, 6.63 |
| imagenet_teddy | dp | 8/15 | 8/8 | 60.1 (54) | 0 | 8/8 | 5.0 (4) | 3 | 92 / 98.91, 40.79 / 100, 4.88 |
| imagenet_teddy | dp-template | 9/15 | 9/9 | 50.9 (44) | 1 | 9/9 | 4.8 (4) | 4 |  |

(fail = failures found / searches on valid seeds; it=1 = failures already at iteration 1; paper = Table 2.)

FP vs DP tests (`runs/main/analysis/*_n15_*.json`): **no significant difference anywhere** (seeds: Fisher p 0.33 /
1.00 / 0.45; iterations: Mann-Whitney p 0.34-1.00, |d| <= 0.25). Diversity still descriptive (<= 20 images per arm):

| task | level | CLIP D: DP / FP | ResNet D: DP / FP |
|---|---|---|---|
| cifar10 | low | 0.253 / 0.259 | 0.262 / 0.300 |
| cifar10 | high | 0.247 / 0.266 | 0.271 / 0.280 |
| pizza | low | 0.176 / 0.155 | 0.199 / 0.155 |
| pizza | high | 0.173 / 0.199 | 0.218 / 0.219 |
| teddy | low | 0.262 / 0.297 | 0.265 / 0.303 |
| teddy | high | 0.266 / 0.273 | 0.296 / 0.302 |

Findings so far (n=15, preliminary):
1. **DP is not more diverse than FP**: DP's D is lower in 9 of 12 comparisons (higher only for pizza at the low
   level). Paper Table 3 claims DP clearly more diverse (and on a ~0.8-0.95 scale vs our ~0.15-0.30, see (10)).
2. **DP loses more seeds** on CIFAR-10 (11/15 vs 14/15; animals "in water") and teddy (8/15 vs 11/15); paper reports DP
   with *more* valid seeds for pizza/teddy.
3. **Iteration-1 failures are common in every arm** (~25-35% of valid seeds): mutants at guidance 1.4 lose the label
   immediately, so these "failures" come from the seed/mutation guidance switch, not from the search. They pull mean
   iterations down equally in all arms. Worth reporting/excluding; ask co-authors.
4. FP vs paper: CIFAR-10 / pizza / teddy FP failure rates and high-level iterations match Table 2 closely
   (e.g. CIFAR high 11.1 vs 14.18, teddy high 5.0 vs 6.63); low-level iterations are lower than the paper (it=1 cases).
5. CelebA FP (our retrained models): low level 9/15 failures (paper 25%), high 15/15 (paper 91%), mean high iterations
   26.9 vs 56.1. Eyeglasses and smiling resist the low step (4 of 6 low-level misses), bangs is easier.
6. dp-template behaves like DP (expected: identical prompts on 58% CIFAR / 28% teddy latents, results reproduce
   exactly when the prompt is identical).

## 2026-11-17 (10): First CIFAR-10 DP results; two fixes; Eq. 6 scale question
- CIFAR-10 DP (5 seeds, same latents as FP): **3/5 seeds valid** (FP 5/5); rejected = "a deer in water", "a frog in
  water". All 3 valid DP seeds fail at both levels. Iterations DP vs FP (low/high): ship 71/9 vs 64/10, ship 189/24 vs
  189/15, automobile 65/9 vs 165/17. Too few to conclude.
- **DP prompt == dp-template prompt** (case-insensitive) for 58/100 CIFAR-10, **100/100 pizza**, 28/100 teddy latents
  (Llama reproduces the template). On those latents the two arms are the same experiment (identical results seen:
  ship 71/9 in both). The LLM-vs-template ablation can only differ where prompts differ; pizza's ablation is void.
  -> `scripts/overnight.sh` now skips dp-template when all prompts in the range are identical (pizza). Relevant to the
  co-authors' framing question (entry 2026-11-13 (3)): with these prompts the LLM adds little beyond the template.
- **Bug fixed (analysis):** with fewer failure images than m=20, every subsample is the whole set -> identical scores
  -> paired t ~1e15, p=0 (nonsense in the first two analysis files, moved to `runs/main/analysis/old_degenerate/`).
  Now: full-set D reported with `insufficient: true` and no test; `stats.paired_t` returns NaN when differences are
  constant. Tests added (50 pass). Also `tests/test_core` delta test updated for delta_scale 2.0 (scaled step equals the
  legacy `init_perturbation` 0.00218342552185059 exactly).
- **Question for co-authors (Table 3 scale):** our Eq. 6 values are ~0.23-0.31 (CLIP and ResNet, CIFAR-10), the paper
  reports ~0.8-0.95. Our code is Eq. 6 literally (mean 1-cos over distinct pairs, verified vs a naive loop). 0.95 would
  mean nearly orthogonal embeddings for same-class images, implausible for CLIP. Ask how Table 3 was computed.
- Pizza seed check: 16/20 = 80% (paper FP 73%).
- Queue restarted at 19:11 without losing work (old bash parent stopped, running pizza FP left to finish; new queue
  waits for it).

## 2026-11-17 (9): CelebA-HQ LoRA trained; seed check 60/60
- LoRA (kohya v21.8.9, recovered recipe): 8550 steps, ~36 min on the 4090, final loss ~0.14. Kept epoch 5 (used) and
  the final epoch-6 file as fallback; epochs 1-4 deleted.
- Seed check (FP prompts, guidance 3.5, 20 per concept): **20/20 smiling, 20/20 bangs, 20/20 eyeglasses = 100% valid**
  (paper 93-94%). Images are clearly faces with the target attribute.
- **Bias to note:** "eye-glasses" seeds are almost all men with sunglasses, many dark-skinned and/or black-and-white
  photo style -- the LoRA amplified the skew of the CelebA eyeglasses subset. "curtain-bangs" seeds are mostly blond women.
  Matters for diversity (Table 3) and for what failures look like; discuss before the paper. Possible fix: balance
  the LoRA training images by Male / Eyeglasses-vs-sunglasses. kohya env + fp32 SD weights kept until decided.
- CelebA FP joins the overnight queue (gate: seed validity >= 0.5).

## 2026-11-17 (8): Disk cleanup (52 -> 70 GB free)
Removed (all re-creatable): `~/data/celeba_packed` (11 GB; rebuild = download flwrlabs/celeba + `dptig.celeba_classifier
prepare`; only needed to retrain the classifier), CelebAMask-HQ zip (extracted images kept in `~/data/celebamask_hq`),
classifier resume checkpoint `last.pt` (final `resnet50_celeba40.pth` + `metrics.json` kept), pip/conda caches,
Miniforge installer. NB `scripts/celeba_replicate.sh` step 1 would now fail if re-run from scratch; steps 3+ skip
because their outputs exist.
Planned once the CelebA seed check passes: drop LoRA epochs other than 5 (and 6 as fallback), kohya env (5.5 GB) and
the fp32 SD weights in the HF cache (only kohya uses them) -- only if no LoRA retraining is needed.

## 2026-11-17 (7): Defaults set from the grid; overnight queue launched
- **Defaults now `delta_scale: 2.0`, `fitness: logit_margin`** in all task configs (= legacy code, closest to
  Table 2; Andrea did not object, "keep crunching"). Reversible: only runs after this point use it; the per-record
  `delta` field shows which. To be confirmed with Maryam (which code produced Table 2).
- CelebA classifier (ours): CelebA test 91.9% mean acc, **CelebA-HQ test 91.3%** (Smiling 93.7, Bangs 96.4,
  Eyeglasses 99.8) -> transfers to the HQ domain. Training 5 epochs x ~130 s.
- LoRA images after the classifier filter: pools 7857 / 1129 / 455 (smiling / bangs / glasses), 38 picked each;
  visually clean. Eyeglasses picks skew male + sunglasses (CelebA bias) -- may show up in what the GA finds.
- `scripts/overnight.sh`: waits for the CelebA pipeline, then round 1 = 5 seeds for every task x {fp, dp, dp-template}
  (+ seed checks for pizza/teddy, CelebA FP if seed validity >= 0.5), then FP and DP grow together to 15/30/50/100
  seeds (dp-template up to 15), with an FP-vs-arm analysis (CPU, diversity on) after each step. CIFAR-10 FP reuses
  the grid's logit_d2 run (identical settings). Output `runs/main/`, progress `runs/main/status.log`.

## 2026-11-17 (6): Phase 1 grid result (CIFAR-10 FP-TIG, 5 seeds x 4 configs, same latents)
No crash in 3.5 h of E-core-pinned running (vs. crashes every 1-25 min on all cores) -> supports the CPU hypothesis.

| config | low: fail | mean (median) it | high: fail | mean (median) it | GPU min |
|---|---|---|---|---|---|
| logit_d1 (default) | 4/5 | 105.4 (106) | 5/5 | 28.4 (22) | 61 |
| logit_d2 (= legacy code) | 5/5 | 93.4 (64) | 5/5 | 9.4 (10) | 48 |
| softmax_d1 (= paper text) | 5/5 | 108.2 (78) | 5/5 | 24.6 (20) | 62 |
| softmax_d2 | 5/5 | 106.6 (84) | 5/5 | 10.8 (12) | 54 |
| **paper FP** | 89.66% | 85.63 | 100% | 14.18 | |

- Seeds valid 5/5 everywhere (paper 87%). Seed 0 (deer) fails at iteration 1 in every config (guidance switch, entry (2)).
- **delta_scale:** high level, d2 needs fewer iterations than d1 for **all 8** non-trivial paired seeds (e.g. auto 17
  vs 82). Paper's 14.18 sits near d2 (9-11), far from d1 (25-28). Low level: no clear effect, all near the paper.
  -> evidence that the paper's numbers come from the legacy code (2x delta).
- **fitness:** logit vs softmax margin indistinguishable at n=5 (same failure criterion, only ranking differs).
- Caveat: 5 seeds; directional only. Waiting on Maryam to confirm which code produced Table 2.

## 2026-11-17 (5): CelebA-HQ replication prepared (runs after the grid)
Andrea asks Maryam for the original CelebA-HQ weights; meanwhile we replicate them so Phase 3 is not blocked.
If her weights arrive, prefer them (Table 2 comparability) and keep ours as a check.
- **LoRA recipe recovered** from the safetensors metadata of the authors' LoRAs: bmaltais/kohya_ss **v21.8.9**
  (commit ed4e3b02), rank 96 / alpha 48, AdamW8bit, lr 1e-4 (UNet) / 5e-5 (TE), cosine, 10% warmup, batch 2, fp16,
  noise offset 0.1, caption dropout 0.1, shuffle caption, 6 epochs, epoch-5 checkpoint used. Pizza/teddy: 38 images
  x 25 repeats, no regularisation; CIFAR-10: 37 x 25 per class + 925 reg. per class. **Captions = trigger phrase only**
  (ss_tag_frequency), not BLIP sentences as the doc says.
- kohya installed at that commit in `~/repos/kohya_ss`, conda env `kohya` (torch 2.0.1+cu118, xformers 0.0.21,
  bitsandbytes 0.41.1, diffusers 0.18.2, transformers 4.30.2; training deps only, no GUI).
- Data (in `~/data`, not in git): official CelebAMask-HQ zip (`liusq/CelebAMask-HQ`, 30k CelebA-HQ images +
  attribute annotations + mapping to CelebA ids; sha256 verified) and `flwrlabs/celeba` (aligned CelebA, official
  split 162770/19867/19962). `huggan/CelebA-faces-with-attributes` was rejected: **11 of its 132 parquet shards are
  truncated on the hub** (sha256 matches the hub, so not a local/CPU problem; deterministic on P- and E-cores).
- New code: `dptig/celeba_classifier.py` (prepare/train; ResNet-50 IMAGENET1K_V1, fc->40, Adam 1e-4 cosine, 5 epochs,
  AMP; eval on CelebA test and on CelebA-HQ images whose source is in the CelebA test split), `dptig/celeba_lora_data.py`
  (38 images/concept, target attribute present and the other two absent, no Blurry, source outside the test split,
  **classifier label-noise filter** p>0.9 / p<0.1: an unfiltered pick contained an "Eyeglasses" face without glasses),
  `scripts/train_celeba_lora.sh` (recovered hyperparameters), `dptig/seed_check.py` (seed validity per class, any task),
  classifier kind `celeba_resnet50`, `configs/celebahq.yaml` (Eq. 4 `target_logit`; Smiling=31, Bangs=5, Eyeglasses=15;
  FP prompts "A photo of <trigger>"; bounds = ImageNet configs).
  `scripts/celeba_replicate.sh` runs it all, idempotent, waits for the grid before using the GPU, pinned to E-cores.
- Tests: `tests/test_celeba.py` (pack/prepare round-trip, classifier kind, selection rules). 48 tests pass.
- **Open:** CelebA DP descriptors (paper: "attribute-specific constraints", not listed) -> pick with BLIP on generated
  seeds, confirm with co-authors; `delta_scale` in celebahq.yaml to follow the grid outcome.

## 2026-11-17 (4): Segfault in pure Python -> suspect CPU (i9-13900KF), grid pinned to E-cores
- After the tgate fix, logit_d1 crashed with `Fatal Python error: Segmentation fault` (faulthandler) inside
  `torch.nn.Module.named_modules` (pure Python, via diffusers `scale_lora_layers`) during a GA batch.
  Together with the empty-closure-cell NameError and the silent deaths: crashes at **different places**, all looking
  like memory corruption. A software bug would usually crash in the same place.
- Machine: **i9-13900KF** (Raptor Lake; known Vmin-shift degradation -> random crashes under load), board Z790 UD AX,
  **BIOS F6 (2023-07-24), predates Intel's fixes**. OS microcode is 0x133 (stops further degradation, cannot undo it).
  No EDAC counters, no access to kernel logs. **Hypothesis, not proven.**
- Test without root: grid relaunched pinned to the E-cores (`PIN="taskset -c 16-31" ./runs_grid.sh`; CPUs 0-15 are
  P-cores). If the crashes stop, that supports the CPU hypothesis. tgate patch kept regardless (bit-identical output).
- Launcher bug fixed: exit codes in attempts.log were always 0 (`$(date)` reset `$?`).
- **For Andrea:** consider a BIOS update (Gigabyte Z790 UD AX, latest with Intel 0x12B+ microcode and "Intel Default
  Settings") and a stress test (e.g. OCCT / y-cruncher / Prime95) to check whether the CPU is degraded.

## 2026-11-17 (3): Grid crashes, TGate fix, robust launcher
- First grid launch died early in every config (logit_d1 after 3 seeds, the others after 1). softmax_d1 died with
  `NameError: free variable 'self' referenced before assignment` inside `tgate/tgate_utils.py` (attention forward),
  at the first `seed_image` after a GA run. logit_d1/logit_d2 stopped with **no traceback** (exit code lost: the old
  launcher piped through grep/tee without pipefail; no access to kernel logs). Not reproducible on demand: 12 rounds of
  seed+3x25 batches ran clean, and a fresh logit_d2 2-seed run completed (seed 1: low F@80, high F@10).
- Cause (best explanation, not proven): tgate 0.1.1 re-wraps every `Attention.forward` in a new closure over `self`
  at every denoising step (module<->closure cycles); occasionally the closure cell is found empty.
- **Fix:** `dptig/tgate_patch.py`: one persistent bound-method forward per Attention module, per-step values stored as
  attributes, same `tgate_processor` and bookkeeping. On by default in `SDGenerator` (`patch_tgate=False` = upstream).
  `tests/test_tgate_patch_gpu.py`: **bit-identical** images vs upstream (seed image + 4-image batch), and checks the
  patched forward is installed. 43 tests pass. Seeds finished before the fix are kept (identical generator output).
- `runs_grid.sh` rewritten: full logs incl. tracebacks, `PYTHONFAULTHANDLER=1`, exit code per attempt in
  `runs/grid/attempts.log`, re-launches a config (resume) up to 3x until it has N records.
- Timing (measured, logit_d1): ~5.4 s per GA generation of 25 images; seed 1 (106+22 gens) 690 s, seed 2 (47+13) 326 s.
- Note: the silent deaths are still unexplained; if `attempts.log` shows non-zero exits with no traceback, suspect
  the OOM killer / GPU memory (23.5 of 24 GB in use during the GA).

## 2026-11-17 (2): First end-to-end CUDA run; Phase 1 grid started
- `dptig.run` CIFAR-10 FP, 1 seed: full path (SD fp16 variant + LoRA + TGate/DeepCache + VGG classifier + GA) runs
  on CUDA with no code changes. Seed 0 ('deer', valid seed) failed at **iteration 1 at both levels** (pred cat,
  margin about -20), 12 s total.
- Seed (guidance 3.5) and mutants (guidance 1.4) differ visibly even at delta=0.001: the guidance switch alone can
  flip the label. Same as legacy (`tig_sd_cifar10.py` l.133/163) and paper Sec. 4.1.1, so not a bug, but it inflates
  iteration-1 failures. **To watch:** fraction of seeds failing at iteration 1 vs paper FP (Iter. 85.63 low / 14.18 high).
- Phase 1 grid launched (`runs_grid.sh`, configs in `configs/grid/`): {logit,softmax}_margin x delta_scale {1,2},
  5 seeds each, output `runs/grid/` (gitignored). Seed 0 of logit_d1 reused from the run above.
  Rough estimate 3-4 h total (unmeasured; based on ~100 generations/seed at a few s each).

## 2026-11-17: New Linux GPU machine set up (RTX 4090, 24 GB)
- Env: miniforge at `~/miniforge3`, conda env `dptig` (Python 3.10), torch 2.3.1+cu121 (driver CUDA 13.2 OK), other
  pins from `requirements-dptig.txt`. `llama-cpp-python` not installed yet (only needed to regenerate prompts).
- 42 tests pass (1 skipped: needs `weights/`).
- **Configs changed:** `base_model` -> `stable-diffusion-v1-5/stable-diffusion-v1-5` (the `runwayml/` repo is gone
  from the hub; same weights). Legacy scripts under `evaluation_results/`/`experiments/` still name `runwayml/`, untouched.
- Base SD 1.5 smoke test on CUDA (fp16, no LoRA, no TGate): 25 images x 50 steps in 23 s (~0.9 s/image), peak 15.6 GB.
  Output looks right. dptig generator/LoRA/classifier path not run yet.
- **Blocked:** `weights/` (LoRAs, classifiers) not on this machine yet; Phase 1 starts once copied.

## 2026-11-15 (3): Resume notes written before moving to the GPU machine (superseded)
State: `main` pushed (7c903a3 + this entry). 42 tests pass in conda env `dptig` (run `python -m unittest discover -s tests`;
pytest is not installed). Dev Mac is for coding/tests only, no experiment runs. Follow `SETUP.md` on the new machine
(copy `weights/`, install `requirements-dptig.txt`; CUDA path untested, expect to fix small things).

Do next, in order:
1. **Phase 1:** CIFAR-10 FP-TIG, `--num-seeds 5`, time one seed (record it here), compare with paper Table 2 FP rows.
   Settle `search.delta_scale` (1.0 vs legacy 2.0) and `fitness` (`logit_margin` vs `softmax_margin`).
2. Run DP and `dp-template` with the same seeds; run `python -m dptig.analyze --fp ... --dp ... --diversity`
   on real output (first real-image test of the CLIP/ResNet extractors and of `analyze`, which is only
   unit-tested on synthetic records).
3. Review prompt diversity: only 40/9/3 distinct DP prompts (CIFAR-10/teddy/pizza); consider higher LLM temperature
   or more descriptors if the DP effect is weak.
4. **Waiting on people:** Maryam (CelebA-HQ classifier + LoRA, details/links for existing weights); co-authors
   (definition of one diversity score, see the Phase 4 entry; wording of `template_phrases` in `configs/`).
5. Remaining Phase 4: MTurk assets/analysis and label-preservation stats (need human data). Then Phase 3, Phase 5.

Unverified so far: anything running Stable Diffusion end-to-end on GPU; real Llama fallback behaviour is fine (0/300).

## 2026-11-15 (2): Phase 4 (partial), diversity metrics + statistics + comparison CLI
- `dptig/metrics.py`: Eq. 6 mean pairwise cosine distance; CLIP ViT-B/32 (HF `openai/clip-vit-base-patch32`, not
  open_clip, to avoid a new dependency) and ResNet-50 (ImageNet V1, fc removed -> 2048-d). `dptig/stats.py`: Fisher,
  Mann-Whitney + Cohen's d, paired t + d_z, Pearson/Spearman. `dptig/analyze.py`: `python -m dptig.analyze --fp DIR --dp DIR [--diversity]`.
- **Assumption (paper silent):** what one diversity "score" is. The paired t-test needs several per condition, so we
  draw k=30 random subsets of m=20 failure-inducing images (same index sets for both arms) and pair by subset. Confirm
  with co-authors; alternatives: per-latent GA population, or one score per run repeat.
- Paired t-test d is d_z (mean diff / SD diff); Mann-Whitney d is pooled-SD Cohen's d. Paper does not specify.
- Verified: 42 unit tests pass (metric properties, naive-loop equivalence, stats, analyze on synthetic records).
  Real CLIP/ResNet extractors smoke-tested on 6 random-noise images: CLIP 512-d, ResNet 2048-d, both load and run on CPU (weights downloaded, ResNet 98 MB). Scores are meaningless on noise; not yet run on real generated images.
- **Not done:** MTurk assets/analysis, label-preservation stats (needs human data), real-data run.

## 2026-11-15: Llama fallback rate measured from the pre-generated prompts (no new runs)
- From `prompts/*.jsonl` (100 each): **0 fallbacks** in all three tasks. First-attempt success: CIFAR-10 78/100
  (9 needed all 5 attempts but still passed), teddy 89/100, pizza 100/100. So the validator + retry works; CIFAR-10
  is the hard case (odd pairs such as "ship on the road").
- **Observation, to review:** few distinct prompts per task: CIFAR-10 40, teddy 9, pizza 3 (per-request seed changes
  sampling but T=0.5 gives near-identical text, esp. for 3 fixed descriptors). Low textual diversity may shrink the
  FP vs DP gap; consider higher temperature / more descriptors if the effect is weak in Phase 5.

## 2026-11-13 (4): Added the LLM-free `dp-template` arm  (commit 21b3d95)
- Prompt = `A photo of a <category> <phrase>`, same target x descriptor combinations as DP-TIG, no LLM, no cache.
  Selectable with `--method dp-template`; records are tagged `dp-template`.
- Motivation: V. Riccio's remark that the LLM step is just a combinatorial application of contextual information. This
  arm is the ablation that tests it (LLM vs. template).
- **My wording, please review:** `template_phrases` in `configs/` (cifar10: "in water" / "on grass" / "on road";
  teddy: "on a bed" / "on a table" / "on a chair"). The paper only gives bare keywords for these. Pizza uses the
  paper's phrases. The LLM arm is unaffected by these.
- Tests: arm is paired with FP/DP (same class/descriptor per latent index) and all prompts pass the validator. 32 tests pass.

## 2026-11-13 (3): Co-author clarifications (via chat, relayed)
- "Merged concept and style LoRA adapters" in Algorithm 1 = just the concept LoRA. No style LoRA is needed
  (no deviation to report).
- CIFAR-10 descriptors (water/grass/road) are sampled uniformly over all 10x3 combinations (Eq. 1), incl. odd pairs.
  Matches the implementation.
- LoRAs/classifiers were trained by Maryam; she should provide details/links. **Open:** do CelebA-HQ weights exist?
- Suggestion to drop the LLM framing in the paper (see entry 4 above).

## 2026-11-13 (2): Phase 2, DP-TIG prompt generation  (commit 6b8a394)
- `dptig/llm_prompt.py`: Fig. 3 instruction text; llama.cpp backend (Llama-3-8B-Instruct-32k Q4_K_M, T=0.5,
  top-p=0.95, per-request seed); constraint validation (starts "A photo of", <12 words, contains category and
  descriptor); retry with fresh sampling seed (max 5); JSONL cache keyed by (category, descriptor, seed);
  lazy model load (GPU box can run from the cache without the LLM).
- `DiversePromptProvider`: class per latent identical to FP's (paired arms); descriptor uniform, deterministic.
- `dptig/gen_prompts.py`: pre-generate prompts without a GPU. `dptig/descriptors.py`: BLIP caption -> ranked
  candidate terms (final shortlist is manual, as in the paper).
- **Choices where the paper is silent:** if the LLM fails all retries, a template prompt is used and flagged
  `llm_fallback: true` in the record. "Under 12 words" is read as <= 11. Pizza descriptors are shown to the LLM as the
  paper's phrases (on a plate / with pepperoni topping / as a slice of pizza); validation checks the keyword.
- **Not verified:** real Llama output quality / fallback rate (model + `llama-cpp-python` not installed yet).

## 2026-11-13 (1): Phase 0, `dptig/` package from the legacy scripts  (commit 6b8a394)
- Weights unpacked to `weights/` (gitignored): `lora/` (MNIST, SVHN, CIFAR-10, pizza, teddy), `classifiers/`
  (CIFAR-10, SVHN). Zips archived in `weights/archive/`. `GENAI_TIGS_WEIGHTS` overrides the location.
- Extracted from `experiments/*/tig_sd_*.py`: GA (`ga.py`), fitness, classifiers (+ exact legacy preprocessing),
  SD/LoRA/TGate wrapper, per-task runner with resume, YAML configs. Legacy code untouched.
- Runner also logs **rejected seeds** (needed for RQ1; the legacy scripts did not).
- **Legacy code vs. paper, defaults follow the paper unless noted (to settle in Phase 1):**
  1. Legacy effectively used delta = 2x (`init_perturbation` = 2x the extent); paper Alg. 1 uses 1x.
     Default 1x. `search.delta_scale: 2.0` reproduces legacy.
  2. Legacy fitness = margin on **raw logits**; paper Eq. 3 says softmax. Default `logit_margin` (matches code);
     `softmax_margin` follows the paper.
  3. Legacy never reset the perturbation step between seeds; we reset per seed (paper).
  4. Legacy `random.choice` was unseeded; class choice is now seeded per latent index.
- Verified: 13 unit tests (fitness equals the legacy function, GA behaviour incl. adaptive delta/clamping/
  determinism, end-to-end runner with fake generator/classifier, CIFAR-10 weights load into the copied VGGNet).
- **Not verified:** anything that needs Stable Diffusion (no GPU / diffusers on the dev machine).

## 2026-11-13 (3): MPS support, Llama smoke test, Mac timing  (commit after fd77450)
- Committed: device-independent RNG (CPU generator for latents/GA noise, so seeds match on CUDA/MPS/CPU; latents are
  NOT bit-identical to the legacy CUDA-generator scripts), configurable SD dtype, MPS sub-batching + attention slicing,
  `fallback_text` (grammatical template phrase if the LLM fails all retries), pinned deps, `prompts/*.jsonl` (100 each).
- Llama smoke test OK: Llama-3-8B-Instruct-32k Q4_K_M in `weights/llm/` loads, 1 prompt in ~6 s, first attempt, no fallback.
  Fallback rate over many prompts still unmeasured. NB `--llm-model` is relative to `weights/`.
- Mac (M-series, 32 GB, MPS) timing, CIFAR-10 FP-TIG, 1 seed: seed image ~10 s, but GA generations of 25 images took
  50-150 s per denoising step (swap ~19/20 GB used, probably memory pressure; unconfirmed). Not usable for real runs.
- **Decision:** no runs on the Mac. Experiments will be run on another machine (set up by Andrea).
- Setup for a new machine written up in `SETUP.md` (env, weights to copy, prompts, run commands). CUDA untested.
- Tests: 33 pass in conda env `dptig` (base python has no torch/diffusers).

## Open questions / next
- Run FP-TIG on the other machine (CIFAR-10 first, `--num-seeds 5`), then compare with Table 2 and settle items 1-2 above.
- Ask Maryam: CelebA-HQ classifier and LoRA weights, and anything else from the original DP-TIG runs.
- ~~Measure the Llama fallback rate~~ done 2026-11-15 (0/300).
- Phase 4: metrics/stats/CLI done; MTurk assets and analysis remain.

Not in the repo: the paper PDF (`docs/`, unpublished preprint).
