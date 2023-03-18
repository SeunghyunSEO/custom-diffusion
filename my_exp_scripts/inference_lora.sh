MODEL_NAME="CompVis/stable-diffusion-v1-4"
SAVE_DIR="cat_lora_r8_alpha16"

CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
    --delta_ckpt logs/$SAVE_DIR/delta.bin \
    --ckpt $MODEL_NAME \
    --prompt "photo of a <new1> cat"  \
    --freeze_model "lora" \
    --lora_r 8 \
    --lora_alpha 16
    # --prompt "A watercolor painting of a <new1> cat" \