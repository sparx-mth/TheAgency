#!/bin/bash
set -e

deepspeed llava/train/train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path liuhaotian/llava-v1.5-13b \
  --version v1 \
  --data_path data/building_maps_dataset.json \
  --image_folder . \
  --output_dir outputs/llava_maps_lora \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --bf16 True --tf32 True \
  --gradient_checkpointing True \
  --lazy_preprocess True \
  --dataloader_num_workers 4 \
  --seed 42 \
  \
  --lora_enable True \
  --lora_r 32 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --mm_projector_lr 1e-5 \
  \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 16 \
  \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --model_max_length 2048 \
  \
  --evaluation_strategy "steps" \
  --eval_steps 50 \
  --save_strategy "steps" \
  --save_steps 50 \
  --save_total_limit 3 \
  --logging_steps 1 \
  --report_to none