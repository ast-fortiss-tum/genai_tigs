# Work log: reimplementing DP-TIG ("Prompt Diversity for Diffusion-Based Test Input Generation")

Newest entries first. Every piece of work adds an entry here: what was done, decisions and deviations from the
paper/legacy code, what was verified vs. not, and what is open.

**Goal.** Reimplement the paper's extension (DP-TIG: LLM-guided target-descriptor prompts + the same SD/GA search,
evaluated against FP-TIG on CIFAR-10, ImageNet pizza/teddy and CelebA-HQ) by reusing this repo's code. The authors'
extension code/models are not available.

**Status at a glance**

| Phase | Scope | State |
|---|---|---|
| 0 | Refactor legacy SD TIG into reusable `dptig/` package (FP-TIG), configs, tests | done, unit-tested; SD path not run (no GPU yet) |
| 1 | Reproduce FP-TIG numbers (paper Table 2, FP rows) on a GPU box | **todo** |
| 2 | DP-TIG: LLM prompts, descriptors, providers, prompt cache | done, tested with stub LLM; real Llama not run yet |
| 2b | LLM-free `dp-template` ablation arm | done |
| 3 | CelebA-HQ (classifier, LoRAs, logit fitness) | todo, waiting to hear from Maryam about existing weights |
| 4 | Diversity metrics (CLIP/ResNet-50), statistics, MTurk assets | todo |
| 5 | Full experiment (100 latents x {FP,DP} x {low,high} x 4 tasks) | todo |

---

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
