import os
import torch
import inspect
import argparse

from tqdm import tqdm
from PIL import Image

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    UNet2DConditionModel,
)

from transformers import CLIPTextModel, CLIPTokenizer
from typing import List, Optional, Tuple, Union


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            # if device != "mps":
            #     logger.info(
            #         f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
            #         f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
            #         f" slighly speed up this function by passing a generator that was created on the {device} device."
            #     )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def get_batched_text_embeddings(tokenizer, text_encoder, prompt, batch_size):
    device = text_encoder.device
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = text_encoder(text_input_ids.to(device))[0]
    bs_embed, seq_len, _ = text_embeddings.shape
    text_embeddings = text_embeddings.repeat(1, batch_size, 1).view(bs_embed * batch_size, seq_len, -1)
    return text_embeddings


def prepare_latents(vae_scale_factor, init_noise_sigma, batch_size,
                    num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (batch_size, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * init_noise_sigma
    return latents


def prepare_extra_step_kwargs(generator, scheduler, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs


def decode_latents(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs="+")
    parser.add_argument("--prompts", type=str, nargs="+")
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scales", type=float, nargs="+")
    parser.add_argument("--eta", type=float, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folder", type=str)
    args = parser.parse_args()

    # load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device).manual_seed(args.seed)

    tokenizers, text_encoders, vaes, unets = [], [], [], []
    noise_scheduler = DDIMScheduler.from_pretrained(args.model_paths[0], subfolder="scheduler")

    for model_path in args.model_paths:
        tokenizers.append(CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer"))
        text_encoders.append(CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").to(device))
        vaes.append(AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device))
        unets.append(UNet2DConditionModel.from_pretrained(model_path, subfolder="unet").to(device))
        print(f"finished loading from {model_path}")

    # sampling
    batch_size = args.bsz
    num_batches = args.num_images // args.bsz
    steps = args.steps
    scales = args.scales
    eta = args.eta
    vae_scale_factor = 2 ** (len(vaes[0].config.block_out_channels) - 1)
    init_noise_sigma = noise_scheduler.init_noise_sigma
    num_channels_latents = unets[0].in_channels
    height = unets[0].config.sample_size * vae_scale_factor
    width = unets[0].config.sample_size * vae_scale_factor
    image_folder = args.folder
    os.makedirs(image_folder, exist_ok=True)

    with torch.no_grad():
        for batch_num in range(num_batches):
            # 1. set the noise scheduler
            noise_scheduler.set_timesteps(args.steps, device=device)
            timesteps = noise_scheduler.timesteps
            # 2. initialize the latents
            latents = prepare_latents(
                vae_scale_factor,
                init_noise_sigma,
                batch_size,
                num_channels_latents,
                height,
                width,
                text_encoders[0].dtype,
                device,
                generator,
                latents=None
            )

            frames = []
            num_warmup_steps = len(timesteps) - steps * noise_scheduler.order
            with tqdm(total=steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

                    uncond_scores, cond_scores = [], []
                    for prompt, tokenizer, text_encoder, unet, vae in zip(args.prompts, tokenizers, text_encoders,
                                                                          unets, vaes):
                        # 3. get the input embeddings
                        text_embeddings = get_batched_text_embeddings(tokenizer, text_encoder, prompt, batch_size)
                        null_embeddings = get_batched_text_embeddings(tokenizer, text_encoder, "", batch_size)
                        input_embeddings = torch.cat((null_embeddings, text_embeddings), dim=0)

                        # predict the noise residual
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=input_embeddings).sample
                        uncond_pred_noise, cond_pred_noise = noise_pred.chunk(2)
                        # save predicted scores
                        uncond_scores.append(uncond_pred_noise)
                        cond_scores.append(cond_pred_noise)

                    # apply compositional score
                    composed_noise_pred = sum(uncond_scores) / len(uncond_scores) + sum(
                        scale * (cond_score - uncond_score) for scale, cond_score, uncond_score in
                        zip(scales, cond_scores, uncond_scores))

                    # compute the previous noisy sample x_t -> x_t-1
                    extra_step_kwargs = prepare_extra_step_kwargs(generator, noise_scheduler, eta)
                    latents = noise_scheduler.step(composed_noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # save intermediate results
                    decoded_images = decode_latents(vae, latents)
                    frames.append(decoded_images)

                    images = decode_latents(vae, latents)
                    images = numpy_to_pil(images)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % noise_scheduler.order == 0):
                        progress_bar.update()

            images = decode_latents(vae, latents)
            images = numpy_to_pil(images)

            for j, img in enumerate(images):
                img_path = os.path.join(image_folder,
                                        f"{args.prompts}_{batch_num * batch_size + j}_{scales}_{args.seed}.png")
                img.save(img_path)


if __name__ == "__main__":
    main()
