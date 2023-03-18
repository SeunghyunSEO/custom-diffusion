MODEL_NAME="CompVis/stable-diffusion-v1-4"
SAVE_DIR="cat_custom_diffusion"

CUDA_VISIBLE_DEVICES=0 python src/sample_diffuser.py \
    --delta_ckpt logs/$SAVE_DIR/delta.bin \
    --ckpt $MODEL_NAME \
    --prompt "photo of a <new1> cat"  \
    --freeze_model "crossattn_kv"
    # --prompt "A watercolor painting of a <new1> cat" \