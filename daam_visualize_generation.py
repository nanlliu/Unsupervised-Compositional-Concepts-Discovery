import torch
import torch.nn.functional as F
import numpy as np
import argparse

from PIL import Image
from daam import trace, set_seed
from matplotlib import pyplot as plt
from diffusers import StableDiffusionPipeline, DDIMScheduler


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--prompt", type=str)
parser.add_argument("--keyword", type=str)
parser.add_argument("--scale", type=float, default=7.5)
parser.add_argument("--num_images", type=int, default=1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe = StableDiffusionPipeline.from_pretrained(args.model_path).to(device)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None
pipe = pipe.to(device)

prompt = args.prompt
keyword = args.keyword
gen = set_seed(0)  # for reproducibility


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def daam_visualize():
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        with trace(pipe) as tc:
            for i in range(args.num_images):
                out, _ = pipe(prompt, num_inference_steps=50, generator=gen, guidance_scale=args.scale)
                img = out.images[0]
                heat_map = tc.compute_global_heat_map()
                heat_map = heat_map.compute_word_heat_map(keyword)
                heat_map.plot_overlay(img, out_file=f"{keyword}_heatmap_{i}.png", word=None)
                img.save(f"{prompt}_{i}.png")


if __name__ == '__main__':
    daam_visualize()
