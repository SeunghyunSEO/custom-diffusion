MODEL_NAME="CompVis/stable-diffusion-v1-4"
SAVE_DIR="cat_lora_r8_alpha16"

accelerate launch src/diffuser_training.py \
    --pretrained_model_name_or_path=$MODEL_NAME  \
    --instance_data_dir=./data/cat  \
    --class_data_dir=./real_reg/samples_cat/ \
    --output_dir=./logs/$SAVE_DIR  \
    --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
    --instance_prompt="photo of a <new1> cat"  \
    --class_prompt="cat" \
    --resolution=512  \
    --train_batch_size=1  \
    --learning_rate=1e-4  \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --num_class_images=200 \
    --scale_lr --hflip  \
    --modifier_token "<new1>" \
    --freeze_model "lora" \
    --lora_r 8 \
    --lora_alpha 16
