# Copyright 2022 Adobe Research. All rights reserved.
# To view a copy of the license, visit LICENSE.md.

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('./')
import torch
from diffusers import StableDiffusionPipeline
from src import diffuser_training 
from PIL import Image

from pdb import set_trace as Tra


def sample(ckpt, delta_ckpt, from_file, prompt, compress, batch_size, freeze_model,
           cones_lr, cones_tau,
           lora_r, lora_alpha):
    
    model_id = ckpt
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    # Tra()

    outdir = 'outputs/txt2img-samples'
    os.makedirs(outdir, exist_ok=True)
    if delta_ckpt is not None:
        diffuser_training.load_model(pipe.text_encoder, pipe.tokenizer, pipe.unet, delta_ckpt, compress, freeze_model,
                                     cones_lr, cones_tau,
                                     lora_r, lora_alpha)
        outdir = os.path.dirname(delta_ckpt)

    all_images = []
    if prompt is not None:
        images = pipe([prompt]*batch_size, num_inference_steps=200, guidance_scale=6., eta=1.).images
        all_images += images
        images = np.hstack([np.array(x) for x in images])
        images = Image.fromarray(images)
        # takes only first 50 characters of prompt to name the image file
        name = '-'.join(prompt[:50].split())
        images.save(f'{outdir}/{name}.png')
    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = [[prompt]*batch_size for prompt in data]

        for prompt in data:
            images = pipe(prompt, num_inference_steps=200, guidance_scale=6., eta=1.).images
            all_images += images
            images = np.hstack([np.array(x) for x in images], 0)
            images = Image.fromarray(images)
            # takes only first 50 characters of prompt to name the image file
            name = '-'.join(prompt[0][:50].split())
            images.save(f'{outdir}/{name}.png')

    os.makedirs(f'{outdir}/samples', exist_ok=True)
    for i, im in enumerate(all_images):
        im.save(f'{outdir}/samples/{i}.jpg')


def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ckpt', help='target string for query',
                        type=str)
    parser.add_argument('--delta_ckpt', help='target string for query', default=None,
                        type=str)
    parser.add_argument('--from-file', help='path to prompt file', default='./',
                        type=str)
    parser.add_argument('--prompt', help='prompt to generate', default=None,
                        type=str)
    parser.add_argument("--compress", action='store_true')
    parser.add_argument("--batch_size", default=5, type=int)
    parser.add_argument('--freeze_model', help='crossattn or crossattn_kv', default='crossattn_kv',
                        type=str)
    
    # for cones
    parser.add_argument("--cones_lr", default=5e-6, type=float)
    parser.add_argument("--cones_tau", default=250, type=float)

    # for lora
    parser.add_argument("--lora_r", type=int, default=8, help="")
    parser.add_argument("--lora_alpha", type=int, default=16, help="")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    sample(args.ckpt, args.delta_ckpt, args.from_file, args.prompt, args.compress, args.batch_size, args.freeze_model, 
           args.cones_lr, args.cones_tau, 
           args.lora_r, args.lora_alpha)