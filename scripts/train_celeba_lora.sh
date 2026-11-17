#!/usr/bin/env bash
# CelebA-HQ concept LoRA ("smiling-face", "curtain-bangs", "eye-glasses"), REPLICATION of the authors' recipe.
# Every hyperparameter below is read from the safetensors metadata of the authors' LoRAs
# (weights/lora/pizza_imgnet-000005.safetensors, cifar10_finetune_lorav1.5-000005.safetensors), trained with
# bmaltais/kohya_ss v21.8.9 (commit ed4e3b02, recorded as ss_sd_scripts_commit_hash). The epoch-5 checkpoint is used.
# Env: conda env "kohya" (torch 2.0.1+cu118, xformers 0.0.21, bitsandbytes 0.41.1, diffusers 0.18.2), ~/repos/kohya_ss.
set -euo pipefail
KOHYA=${KOHYA:-$HOME/repos/kohya_ss}
PY=${PY:-$HOME/miniforge3/envs/kohya/bin/python}
DATA=${DATA:-$HOME/data/celeba_lora_train/img}        # built by: python -m dptig.celeba_lora_data
OUT=${OUT:-$(cd "$(dirname "$0")/.." && pwd)/weights/lora}
BASE=${BASE:-$(ls -d $HOME/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots/*/ | head -1)}
NAME=celebahq_lorav1.5

# steps: images x repeats / batch x epochs; warmup = 10% (authors: 285/2850)
n_img=$(find "$DATA" -name '*.png' | wc -l)
steps_per_epoch=$(( (n_img * 25 + 1) / 2 ))
warmup=$(( steps_per_epoch * 6 / 10 ))
echo "images=$n_img steps/epoch=$steps_per_epoch warmup=$warmup base=$BASE"

cd "$KOHYA"
$PY train_network.py \
  --pretrained_model_name_or_path "$BASE" \
  --train_data_dir "$DATA" \
  --output_dir "$OUT" --output_name "$NAME" --save_model_as safetensors --save_precision fp16 \
  --logging_dir "$OUT/../lora_logs" \
  --resolution 512,512 --enable_bucket --min_bucket_reso 256 --max_bucket_reso 2048 --bucket_no_upscale \
  --network_module networks.lora --network_dim 96 --network_alpha 48 \
  --learning_rate 1e-4 --unet_lr 1e-4 --text_encoder_lr 5e-5 \
  --lr_scheduler cosine --lr_warmup_steps "$warmup" \
  --optimizer_type AdamW8bit --max_grad_norm 1.0 \
  --train_batch_size 2 --max_train_epochs 6 --save_every_n_epochs 1 \
  --mixed_precision fp16 \
  --noise_offset 0.1 --caption_dropout_rate 0.1 --shuffle_caption --prior_loss_weight 1.0 \
  --max_data_loader_n_workers 4 --xformers --seed 1234 \
  "$@"
