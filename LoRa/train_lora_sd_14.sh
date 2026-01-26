export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="LoRa/playdough/shape1/playdough_shape1_dataset"
export OUTPUT_DIR="LoRa_playdough_shape1"

accelerate launch /home/sol/diffusers/examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --dataloader_num_workers=4 \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  --checkpointing_steps=500 \
  --validation_prompt="" \
  --seed=1337