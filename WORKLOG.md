# DP-TIG reimplementation: current state (2026-11-27)

Reimplementation of the paper's extension ("Prompt Diversity for Diffusion-Based Test Input Generation"): DP-TIG
(LLM target-descriptor prompts) vs FP-TIG (fixed prompt), same SD 1.5 + GA search, on CIFAR-10, ImageNet pizza,
ImageNet teddy bear and CelebA-HQ. Code in `dptig/`, results in `runs/main/`.

This file keeps only what is current. The full chronological log (every decision, bug fix and intermediate result)
is in [docs/worklog_history.md](docs/worklog_history.md); "entry (n)" below refers to it. New work: add a dated
entry at the end of this file and update the sections above it.

## 1. Where we are

- **Done:** all 4 tasks x 3 arms (FP, our DP, Maryam's DP) at **30 seeds**, analysed (section 3); CIFAR-10, pizza and
  teddy at **50 seeds**, analysed (section 3b).
- **Running:** `scripts/queue_weekend.sh` (replaced `queue_100.sh` on 2026-11-27 16:45, entry 2026-11-27 (23) in
  the log below): 0 CelebA analyses at 50 + pre-registered diversity test; A new arm `dp-paper-nostyle` to 50;
  B CIFAR-10 / pizza / teddy to the paper's 100 seeds; C nostyle to 100; D CelebA to 100. Each phase regenerates,
  commits and pushes [docs/auto_results.md](docs/auto_results.md) (machine-written tables).
- **Not started:** RQ4 / RQ5 (human evaluation), see section 5.

## 2. Setup

**Arms** (all paired: seed n uses the same initial latent and the same class/attribute in every arm)
| Arm | Prompt | SD weights |
|---|---|---|
| `fp` | fixed per class, e.g. "A photo of 1toy_bear teddy_bear" | concept LoRA |
| `dp` (ours) | Llama-3-8B, our instruction from the paper's Fig. 3: class + randomly sampled descriptor (e.g. "A photo of a teddy bear sitting on a chair") | concept LoRA (as the paper text and the co-authors' answer of 2026-11-13 say) |
| `dp-paper` (Maryam's code) | her per-task LLM instruction verbatim | concept LoRA **merged with a style LoRA** (CIFAR-10 0.7/0.3, pizza 0.5/0.5, teddy 0.01/0.3) |

Prompts are generated once and cached (`prompts/*.jsonl`, 100 per task and arm). A fourth arm, `dp-template`
(descriptor inserted by a template, no LLM), was run at 15 seeds on CIFAR-10 and teddy only; it behaves like `dp`.

**Settings** (match the code that produced Table 2, confirmed by Maryam's scripts; entries (6), (16))
- SD 1.5, 512x512, 25 steps, TGate; guidance 3.5 for the seed image, 1.4 for mutants.
- GA: population 25, 10 best kept, max 250 iterations; perturbation = latent range / 10000 (low) or / 1000 (high),
  **times 2** (legacy code; the paper text says x1); fitness = raw-logit margin (paper text: softmax margin); step x2
  on stall, latents clamped to bounds.
- CelebA: Maryam's LoRA, our ResNet-50 classifier (equivalent to hers on our seeds), fitness = target-attribute
  logit, failure = attribute predicted absent. Her CelebA GA differs (no clamping, x0.75 on stall, bounds/deltas
  from a `celeb_sd_params` file that is not in her repo); we kept ours.
- Concept LoRAs and the CIFAR-10 classifier are byte-identical to Maryam's.

**Diversity (RQ6)** = Maryam's Table-3 metric, which is *not* the paper's Eq. 6: KMeans on the union of
FP-low/FP-high/DP-low/DP-high embeddings for K = 2, 4, ..., 12; per group, coverage x normalised entropy of the
cluster histogram; FP vs DP by paired t-test over the 6 K values. Embeddings of the classifier input (CIFAR 32x32,
ImageNet 224 crop). **ResNet-50 is the reported feature space** (Andrea, 2026-11-24), CLIP secondary. We repeat the
clustering with 10 KMeans seeds and report how often DP > FP and how often p < 0.05 (the result depends on it, 4.1).

**Statistics:** RQ1/RQ2 Fisher exact; RQ3 Mann-Whitney U + Cohen's d on iterations of searches that found a failure.

## 3. Results at 30 seeds

Paper values: Table 2 (DP vs FP, 100 seeds). Ours: FP / DP / dp-paper. Iterations = mean over failing searches.
p-values: FP vs DP, FP vs dp-paper.

**RQ1 seed validity (% of seeds the classifier labels correctly)**
| Task | Paper DP / FP | Ours FP / DP / dp-paper | p |
|---|---|---|---|
| CIFAR-10 | 84 / 87 | 93 / 83 / 83 | 0.42 / 0.42 |
| Pizza | 85 / 73 | 77 / 60 / 83 | 0.27 / 0.75 |
| Teddy | 92 / 87 | 93 / **50** / 97 | **0.0004** / 1.0 |
| CelebA | 93 / 94 | 100 / 100 / 100 | 1 / 1 |

**RQ2 effectiveness (% of valid seeds with a failure), low level** (high level: 100% everywhere, CelebA paper 96 / 91)
| Task | Paper DP / FP | Ours FP / DP / dp-paper | p |
|---|---|---|---|
| CIFAR-10 | 92.9 / 89.7 | 89 / 88 / 84 | n.s. |
| Pizza | 88.2 / 97.3 | 100 / 100 / 100 | 1 |
| Teddy | 98.9 / 98.9 | 100 / 100 / 100 | 1 |
| CelebA | 18.3 / 24.7 | 57 / 67 / 67 | 0.60 |

**RQ3 iterations to first failure, low level** (high level: same directions, all n.s. except pizza dp-paper p=0.05)
| Task | Paper DP / FP | Ours FP / DP / dp-paper | p (d) |
|---|---|---|---|
| CIFAR-10 | 74.7 / 85.6 | 77.9 / 68.5 / 65.2 | 0.50 / 0.53 |
| Pizza | 123.3 / 83.6 | 66.1 / 56.9 / **121.1** | 0.83 / **0.009 (d=0.80)** |
| Teddy | 40.8 / 48.5 | 56.3 / 45.2 / 48.6 | 0.50 / 0.29 |
| CelebA | 8.8 / 18.2 | 125.6 / 127.2 / 140.8 | 0.94 / 0.60 |

**RQ6 diversity, Table-3 metric, ResNet-50, overall (low + high).** D at KMeans seed 0 (FP -> DP), then "DP > FP in
a/10 KMeans seeds, significant in b/10"
| Task | FP vs DP | FP vs dp-paper |
|---|---|---|
| CIFAR-10 | 0.93 -> 0.92, 1/10, 0/10 (high: 0.93 -> 0.82, 0/10) | 0.96 -> 0.97, 3/10, 0/10 |
| Pizza | 0.68 -> 0.73, 7/10, 1/10 | 0.60 -> 0.53, 6/10, 1/10 |
| Teddy | **0.72 -> 0.91, 9/10, 3/10** (low: 10/10, sig 5/10) | 0.88 -> 0.88, 8/10, 0/10 |
| CelebA | 0.92 -> 0.93, 9/10, 2/10 (low: 10/10, sig 5/10) | 0.93 -> 0.91, 0/10, 1/10 |

CLIP: the only robust effect is pizza dp-paper, low level (0.50 -> 0.75, significant in 10/10). Analysis files:
`runs/main/analysis/<task>_fp_vs_<arm>_n30_*.json` (newest per pair).

**What this says**
- **FP reproduces Table 2** on CIFAR-10, pizza and teddy (e.g. CIFAR-10 low 89% / 78 it vs 89.66% / 85.63). The
  baseline is sound; differences come from DP.
- **Neither DP setup reproduces "DP is as valid and more diverse".** Our DP is more diverse on teddy (and slightly
  on CelebA low) but loses seeds, significantly on teddy. Maryam's DP keeps its seeds but behaves like FP; it
  reproduces the paper's one large RQ3 effect (pizza, DP much slower: 121 vs 66 it; paper 123 vs 84).
- **RQ2 agrees** with the paper on CIFAR-10 / pizza / teddy (~100% everywhere).
- **CelebA does not match the paper's FP numbers**: we find failures far more often at the low level (57-67% vs
  18-25%) but need ~7x more iterations (126 vs 18). Most likely the GA differences above (clamping, step policy,
  unknown `celeb_sd_params`); not resolvable without that file.
- No significant result at 30 seeds is in the paper's favour except pizza RQ3 with dp-paper.

## 3b. Results at 50 seeds (CIFAR-10, pizza, teddy; CelebA still running)

Same layout as section 3 (FP / DP / dp-paper; p: FP vs DP, FP vs dp-paper). Files:
`runs/main/analysis/<task>_fp_vs_<arm>_n50_*.json`, per-arm clustering `runs/main/analysis/<task>_t3_separate_n50.json`.

**RQ1 seed validity (%)**
| Task | Ours FP / DP / dp-paper | p |
|---|---|---|
| CIFAR-10 | 94 / 86 / 90 | 0.32 / 0.72 |
| Pizza | 78 / 58 / 86 | 0.053 / 0.44 |
| Teddy | 90 / **54** / 94 | **0.0001** / 0.72 |

**RQ2 effectiveness, low level (%)** (high level: 100% everywhere)
| Task | Ours FP / DP / dp-paper | p |
|---|---|---|
| CIFAR-10 | 91 / 88 / 87 | 0.73 / 0.52 |
| Pizza | 97 / 100 / 95 | 1 / 1 |
| Teddy | 100 / 100 / 100 | 1 / 1 |

**RQ3 iterations to first failure, low level**
| Task | Ours FP / DP / dp-paper | p (d) |
|---|---|---|
| CIFAR-10 | 72 / 63 / 60 | 0.36 / 0.25 |
| Pizza | 74 / 73 / **121** | 0.89 / **0.006 (d=0.70)** |
| Teddy | 54 / 34 / 38 | 0.09 (d=-0.45) / 0.055 (d=-0.33) |

High level: teddy DP and dp-paper are significantly *faster* than FP (p=0.047, d=-0.58; p=0.017, d=-0.29), the
opposite of the paper's direction; pizza dp-paper slower (p=0.07).

**RQ6 diversity, Table-3 metric, ResNet-50, overall.** Mean D over 10 KMeans seeds (FP -> arm), "arm > FP in a/10
seeds, significant (in either direction) in b/10". *Joint* = authors' procedure (KMeans on FP + arm pooled);
*per-arm* = each arm clustered on its own (`scripts/t3_separate.py`).
| Task / arm | Joint | Per-arm |
|---|---|---|
| CIFAR-10 DP | 0.95 -> 0.95, 3/10, 1/10 | 0.95 -> 0.95, 6/10, 0/10 |
| CIFAR-10 dp-paper | 0.96 -> 0.96, 6/10, 2/10 | 0.95 -> 0.95, 4/10, 0/10 |
| Pizza DP | 0.87 -> 0.84, 2/10, 1/10 | 0.85 -> 0.79, 0/10, 2/10 |
| Pizza dp-paper | **0.47 -> 0.71, 10/10, 10/10** | 0.85 -> 0.73, **0/10**, 1/10 |
| Teddy DP | 0.80 -> 0.87, 9/10, 2/10 | 0.86 -> 0.80, **0/10**, 0/10 |
| Teddy dp-paper | 0.86 -> 0.85, 3/10, 1/10 | 0.86 -> 0.88, 7/10, 2/10 |

CLIP, per-arm: the only case where DP is consistently higher is CIFAR-10 (DP 10/10, sig 1/10; dp-paper 10/10, sig
5/10, 0.94 -> 0.97). Pizza dp-paper joint CLIP (FP 0.71 -> 0.85, sig 8/10) also drops to 0.92 -> 0.86 per-arm.

**What this says**
- RQ1-RQ3 at 50 seeds confirm the 30-seed picture: our DP loses seeds (teddy, nearly pizza), dp-paper does not; RQ2
  ~100% in all arms; the only paper-direction RQ3 effect is pizza dp-paper (121 vs 74, paper 123 vs 84), now
  stronger; teddy goes the other way.
- **The one strong diversity result (pizza dp-paper, 10/10 significant) is an artefact of the joint clustering.** The
  same FP images score 0.87 when clustered with DP and 0.47 when clustered with dp-paper. dp-paper's style LoRA
  gives its images a distinct look, so the pooled KMeans partly separates the arms and FP lands in few clusters.
  Clustered separately, dp-paper is *less* diverse than FP (0.73 vs 0.85). Teddy DP's joint advantage also
  reverses per-arm. See issue 8.

## 4. Issues for the paper (co-authors)

1. **Table-3 diversity test is not robust.** Same images, different KMeans seed: p from 0.01 to 0.97 and the
   direction flips (CIFAR-10: DP > FP in 8/10, 0/10 or 3/10 seeds depending on level). Even image order changes the
   result. The t-test pairs 6 K values computed on the same images, so they are not independent samples and p is
   overstated. Recommendation: report a bootstrap CI over images, or Eq. 6 with independent subsamples, and state
   the clustering seed.
2. **Style-LoRA confound in Maryam's DP.** Her DP runs merge a style LoRA that FP does not use (teddy: concept LoRA at
   0.01, i.e. practically off), so FP vs DP differ in prompt *and* weights. The paper and the co-authors' answer of
   2026-11-13 say there is no style LoRA. Our `dp` arm is the clean comparison.
3. **The LLM adds little.** Maryam's instructions dictate the prompt ("A photo of a pizza {style_phrase}"): 3 distinct
   prompts for pizza and teddy. Our LLM prompts equal the plain template for 100% (pizza), 58% (CIFAR-10) and 28%
   (teddy) of latents. The contribution is the descriptors, not the LLM.
4. **Why our DP loses seeds (RQ1).** We regenerated the 18 rejected teddy seeds (`runs/main/invalid_seeds/`): almost
   all are recognisable teddy bears; the classifier labels the furniture from the descriptor (rocking chair, dining
   table, crate) or reads realistic fur as a dog breed. A classifier-validity effect, not bad images; relevant to how
   RQ1 is argued.
5. **Algorithm 1 / Eq. 3 vs the code.** Table 2 comes from delta x2 and a raw-logit margin (our 2x2 grid, entry (6),
   and Maryam's scripts), not x1 and softmax as written. The text should be fixed.
6. **Iteration-1 failures.** The first mutant generation is often already misclassified, in every arm: ~20-35% of
   searches on CIFAR-10, up to ~50% at the ImageNet high level (teddy dp-paper 15/29), ~5% on CelebA. Cause: mutants
   use guidance 1.4, the seed 3.5. This lowers mean iterations for all arms. Exclude, report
   separately, or generate seeds at 1.4?
7. **Doc typo:** `documentation/sd_ft_train_param_settings.md` gives the teddy trigger as `1teddy_toy`; the LoRA was
   trained with `1toy_bear`. We used the wrong one until 2026-11-24 (old FP archived, all reported results use the
   correct one).
8. **Table-3 joint clustering measures how different the arms look, not only diversity.** KMeans is fit on FP and
   DP images pooled (Maryam's `compute_diversity_*.py`), so each arm's score depends on the other arm: if DP images
   differ systematically (style LoRA, descriptors such as furniture), clusters split by arm and FP's score drops.
   At 50 seeds this alone produces pizza dp-paper 0.47 -> 0.71 (10/10 significant); with per-arm clustering the
   effect is gone (0.85 -> 0.73). Recommendation: cluster each arm separately (or on a fixed reference set) and
   report that. Section 3b.

## 5. Next

1. **100 seeds** (running). 50 seeds done for CIFAR-10 / pizza / teddy (2026-11-27 08:50); CelebA at 50 expected
   ~2026-11-28; all at 100 ~2026-12-01. Rerun `scripts/t3_separate.py` at 100 seeds next to the joint analysis.
2. **RQ4 / RQ5 (human study):** Maryam's repo contains the original MTurk batches and answers
   (`evaluation_results/mturk-evaluation/`), so these RQs can be reanalysed from existing data instead of a new study.
   Not done yet.
3. **Optional extension: modern generator.** Does FP vs DP hold with SD 3.5 Medium (or Flux-schnell)? Not a
   replacement: the replication must stay on SD 1.5 to be comparable. Needs no concept LoRAs (plain prompts), a new
   calibration of perturbation/guidance (different latent space and sampler), ~3-5x slower per image on the 4090.
   Plan: teddy + pizza, 30 seeds, after the 100-seed run; ~2-4 days.

## 6. Reproduce / operate

- Env: conda `dptig` (`~/miniforge3`); GPU box RTX 4090 24 GB. 56 tests: `python -m unittest discover -s tests`.
- **Pin long runs to E-cores** (`taskset -c 16-31`): the i9-13900KF crashes on P-cores (BIOS 2023); one hard crash on
  2026-11-25 even when pinned. A BIOS update + stress test is recommended.
- One run: `python -m dptig.run --config configs/<task>.yaml --method {fp,dp,dp-paper} --out runs/main/<task>_<arm> --num-seeds N`
  (resumes from `records.jsonl`).
- One comparison: `python -m dptig.analyze --fp runs/main/<task>_fp --dp runs/main/<task>_<arm> --diversity --table3 --view {cifar10,imagenet,celeba} --kmeans-seeds 10 --device cpu`.
- Summary table: `python -m dptig.summarize runs/main`.
- Queue: `setsid nohup ./scripts/queue_weekend.sh >> runs/main/queue_weekend.out 2>&1 < /dev/null &` (resumes; restart the same
  way if the machine goes down). Progress: `runs/main/status.log`. **Do not change configs or generation code while
  it runs**, or seeds 30-99 are not comparable with 0-29.
- Maryam's code: https://github.com/deeptestai/genai_tigs_diversity (local clone `~/repos/maryam_src/`).

## Log (new entries here)

- **2026-11-26** WORKLOG condensed to the current state for review; full history moved to
  `docs/worklog_history.md` (entries (1)-(22)). `scripts/queue_100.sh` running.
- **2026-11-27** 50-seed results for CIFAR-10, pizza and teddy (section 3b). Per-arm Table-3 clustering
  (`scripts/t3_separate.py`, CPU) shows the pizza dp-paper diversity effect is an artefact of the joint clustering
  (issue 8). CelebA at 50 running.
- **2026-11-27 (23)** Plan until 2026-12-02, `scripts/queue_weekend.sh` replaces `queue_100.sh` (stopped by hand;
  its CelebA dp-paper run to 50 was left to finish). Two additions:
  1. **New arm `dp-paper-nostyle`** (`configs/<task>_nostyle.yaml`): dp-paper prompts (same cache, same latents) on
     FP's weights (concept LoRA at 1.0, no style LoRA). Separates the prompt effect from the style-LoRA confound
     (issue 2). CIFAR-10 / pizza / teddy only (CelebA dp-paper has no style LoRA).
  2. **Pre-registered RQ6 test** (`scripts/div_perm.py`), fixed now, before the results it produces are read, and
     the only diversity test used to answer "is DP more diverse?" from here on. Primary: Eq. 6 (mean pairwise
     cosine distance), ResNet-50 V2 features of the classifier input, failure images of valid seeds, per level,
     each arm vs FP; two-sided permutation test on D_arm - D_FP (10k), 95% bootstrap CI over images; Holm per arm
     over all task x level tests. Secondary: same with CLIP. Replaces the paired t over overlapping subsamples
     (Eq. 6 above) and over the 6 K values (Table 3), whose samples are not independent (issue 1). (A smoke test on
     teddy was run while writing the script; phase 0 recomputes it unchanged.)
  - **Correction (RQ4/RQ5):** the MTurk study in Maryam's repo asked workers for the category / attributes of the
    paper's images, i.e. validity and label preservation. It says nothing about diversity, so it cannot support
    RQ6.
