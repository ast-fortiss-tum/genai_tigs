# Setting up DP-TIG on a new machine

Verified on macOS (Apple Silicon, MPS) only. A CUDA machine has not been tested yet; the code picks the device
automatically (`pick_device` in `dptig/generator.py`) and uses fp16 on CUDA, fp32 elsewhere.
**Do not run experiments on the dev Mac**: a GA generation took 50-150 s per denoising step (see WORKLOG.md, entry 3).

## 1. Environment
```
git clone <repo> && cd genai_tigs
conda create -n dptig python=3.10 && conda activate dptig
pip install -r requirements-dptig.txt
```
- The pins in `requirements-dptig.txt` come from the macOS env. On CUDA, install the matching torch/torchvision
  wheels first (keep torch 2.3.x / diffusers 0.27.2 / peft 0.10.0 / tgate 0.1.1 / DeepCache 0.1.1 to match the legacy code).
- `llama-cpp-python` is only needed to *generate* DP prompts, not to run them. CUDA build:
  `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`.
- Tests: `python -m unittest discover -s tests` (42 pass on the dev Mac).

## 2. Weights (not in git; `weights/` is gitignored)
Copy `weights/` from the dev Mac (about 840 MB without the LLM), or set `GENAI_TIGS_WEIGHTS=/path`:
```
weights/lora/        cifar10_finetune_lorav1.5-000005.safetensors, svhn_..., Mnist_..., pizza_imgnet-000005..., teddy_bear_imagenet-000005...
weights/classifiers/ cifar10/, svhn/
weights/llm/         Llama-3-8B-Instruct-32k-v0.1.Q4_K_M.gguf   (4.9 GB; optional, see below)
```
- Original zips are in `weights/archive/` on the dev Mac.
- Missing: CelebA-HQ LoRA and classifier (asked of Maryam), ImageNet classifiers (torchvision pretrained, not checked).
- Stable Diffusion: `runwayml/stable-diffusion-v1-5` is downloaded from the Hugging Face hub on first use (about 4 GB).

## 3. Prompts
`prompts/{cifar10,imagenet_pizza,imagenet_teddy}_dp.jsonl` (100 each) are committed, so DP-TIG runs **without** the
LLM or the GGUF. To regenerate or extend: `python -m dptig.gen_prompts --config configs/cifar10.yaml --num 100
--llm-model llm/<file>.gguf` (path is relative to `weights/`).

## 4. Running
```
python -m dptig.run --config configs/cifar10.yaml --method fp          --out runs/cifar10_fp --num-seeds 5
python -m dptig.run --config configs/cifar10.yaml --method dp          --out runs/cifar10_dp --num-seeds 5
python -m dptig.run --config configs/cifar10.yaml --method dp-template --out runs/cifar10_dpt --num-seeds 5
```
- Runs resume from `--out`. Output includes seed images, and rejected seeds (needed for RQ1).
- `--num-seeds` defaults to the config's 100. Start with 5 and compare to Table 2 of the paper.
- First thing to settle on the new machine: `search.delta_scale` (1.0 paper vs 2.0 legacy) and `fitness`
  (`logit_margin` legacy vs `softmax_margin` paper); see WORKLOG.md entry 1.
- Time one seed first and record it in WORKLOG.md (expected: minutes per seed on a modern GPU; unmeasured).

## 5. Analysis (Phase 4)
```
python -m dptig.analyze --fp runs/cifar10_fp --dp runs/cifar10_dp --diversity --device cuda
```
Needs no extra install: CLIP ViT-B/32 comes from `transformers` (HF `openai/clip-vit-base-patch32`) and ResNet-50
from torchvision; both download on first use (about 100 MB for ResNet). Use `--k/--m` to change the diversity subsampling (default 30 x 20).
