MODEL_NAME="CompVis/stable-diffusion-v1-4"
SAVE_DIR="cat_cones_attn2_only_reproduce"

CUDA_VISIBLE_DEVICES=1 python src/sample_diffuser.py \
    --delta_ckpt logs/$SAVE_DIR/delta.bin \
    --ckpt $MODEL_NAME \
    --prompt "photo of a <new1> cat"  \
    --freeze_model "cones" \
    --cones_lr 2e-5 \
    --cones_tau 3200
    # --prompt "A watercolor painting of a <new1> cat" \