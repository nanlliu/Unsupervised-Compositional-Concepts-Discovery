import os
import torch
import argparse

from diffusers import StableDiffusionPipeline, DDIMScheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--prompts", type=str, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", type=str, default="7.5")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--folder_name", type=str, default="samples")
    args = parser.parse_args()

    model_id = args.model_path
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None

    folder = os.path.join(args.model_path, args.folder_name)
    os.makedirs(folder, exist_ok=True)

    generator = torch.Generator("cuda").manual_seed(args.seed)
    prompts = args.prompts
    weights = args.weights

    if prompts:
        batch_size = args.bsz
        num_batches = args.num_images // batch_size
        for prompt in prompts:
            for i in range(num_batches):
                image_list = pipe(prompt, num_inference_steps=50, guidance_scale=args.scale,
                                    generator=generator, num_images_per_prompt=batch_size)
                images = image_list.images
                for j, img in enumerate(images):
                    img.save(os.path.join(folder, f"{prompt}_{weights}_{args.seed}_{i * batch_size + j}.png"))
