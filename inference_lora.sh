#!/bin/bash
# FLUX ControlNet LoRA Inference for SPAD→RGB

source ~/miniconda3/etc/profile.d/conda.sh
conda activate diffsynth

# Paths
LORA_CHECKPOINT="./models/train/FLUX-SPAD-ControlNet-LoRA/epoch-9.safetensors"
SPAD_IMAGE="/home/jw/engsci/thesis/spad/spad_dataset/bits/0724-dgp-001_RAW_empty_frames0-0_p.png"
OUTPUT="./output_rgb.png"

# Run inference
python inference_lora.py \
  --lora_checkpoint "${LORA_CHECKPOINT}" \
  --lora_target controlnet \
  --control_image "${SPAD_IMAGE}" \
  --output "${OUTPUT}" \
  --prompt "" \
  --height 512 \
  --width 512 \
  --steps 20 \
  --cfg_scale 1.0 \
  --embedded_guidance 3.5 \
  --controlnet_scale 1.0 \
  --seed 42

echo ""
echo "Inference complete! Output: ${OUTPUT}"
