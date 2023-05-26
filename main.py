import argparse
import itertools
import math
import os
import random
import json

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

import PIL
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DDIMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

from datasets import ComposableDataset

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_id, accelerator, args):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {args.placeholder_tokens: learned_embeds.detach().cpu()}
    if args.test:
        embed_path = os.path.join(args.output_dir, "test_learned_embeds.bin")
    else:
        embed_path = os.path.join(args.output_dir, "learned_embeds.bin")
    torch.save(learned_embeds_dict, embed_path)


def save_weights(weights, args):
    logger.info("Saving embeddings")
    learned_weights_dict = {"weights": weights.detach().cpu()}
    if args.test:
        weight_path = os.path.join(args.output_dir, "test_weights.bin")
    else:
        weight_path = os.path.join(args.output_dir, "weights.bin")
    torch.save(learned_weights_dict, weight_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, required=True,
        help="A list of folders containing the training data for each token provided."
    )
    parser.add_argument(
        "--placeholder_tokens",
        type=str,
        required=True,
        help="A list of tokens to use as placeholders for all the concepts, separated by comma",
    )
    parser.add_argument(
        "--initializer_tokens", type=str, default="",
        help="A list of tokens to use as initializer words, separated by comma"
    )
    parser.add_argument("--learnable_property", type=str, default="",
                        help="a list of properties for all the tokens needed to be learned, separated by comma")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default="",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--softmax_weights", action="store_true", default=False)
    parser.add_argument("--reuse_weights", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--use_composed_score", action="store_true", default=False,
        help="whether to use composed score for textual inversion."
    )
    parser.add_argument(
        "--use_orthogonal_loss", action="store_true", default=False,
        help="should be enabled to get a better performance when using composed scores to invert text."
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )

    parser.add_argument("--data", type=str, default="imagenet")
    parser.add_argument("--class_folder_names", type=str,
                        help="a list of imagenet data folders for each class, seperate by comma")

    parser.add_argument("--add_weight_per_score", action="store_true", default=False)
    parser.add_argument("--freeze_weights", action="store_true", default=False)
    parser.add_argument("--init_weight", type=float, default=1)
    parser.add_argument("--use_conj_score", action="store_true", default=False)

    parser.add_argument("--orthogonal_coeff", type=float, default=0.1)
    parser.add_argument("--squared_orthogonal_loss", action="store_true", default=False)
    parser.add_argument("--mse_coeff", type=float, default=1)
    parser.add_argument("--num_images_per_class", type=int, default=-1, help="-1 means all images considered")

    parser.add_argument("--weighted_sampling", action="store_true", default=False)
    parser.add_argument("--flip_weights", action="store_true", default=False)

    parser.add_argument("--text_loss", action="store_true", default=False)
    parser.add_argument("--text_angle_loss", action="store_true", default=False)
    parser.add_argument("--text_repulsion_loss", action="store_true", default=False)
    parser.add_argument("--text_repulsion_similarity_loss", action="store_true", default=False)
    parser.add_argument("--text_repulsion_coeff", type=float, default=0)

    parser.add_argument("--euclidean_dist_loss", action="store_true", default=False)
    parser.add_argument("--euclidean_dist_coeff", type=float, default=0)

    parser.add_argument("--use_similarity", action="store_true", default=False,
                        help="Dot product between scores as the orthogonal loss")
    parser.add_argument("--use_euclidean_mhe", action="store_true", default=False,
                        help="Minimum Hyperspherical Energy as the orthogonal loss")
    parser.add_argument("--log_mhe", action="store_true", default=False)
    parser.add_argument("--use_acos_mhe", action="store_true", default=False)
    parser.add_argument("--normalize_score", action="store_true", default=False)
    parser.add_argument("--use_weighted_score", action="store_true", default=False)

    parser.add_argument("--use_l2_norm_regularization", action="store_true", default=False)
    parser.add_argument("--l2_norm_coeff", type=float, default=0)

    parser.add_argument("--normalize_word", action="store_true", default=False)
    parser.add_argument("--num_iters_per_image", type=int, default=50)
    parser.add_argument("--hsic_loss", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False,
                        help="enable this by only optimizing weights using existing models.")

    parser.add_argument(
        "--validation_step",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )


    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.resume_from_checkpoint:
        args.pretrained_model_name_or_path = args.resume_dir
        print(f"resume everything from {args.pretrained_model_name_or_path}")

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Add the placeholder token in tokenizer
    placeholder_tokens = [x.strip() for x in args.placeholder_tokens.split(",")]
    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)

    if num_added_tokens != 0 and num_added_tokens != len(placeholder_tokens):
        raise ValueError(
            f"The tokenizer already contains at least one of the tokens in {placeholder_tokens}. "
            f"Please pass a different placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    if args.initializer_tokens != "":
        initializer_tokens = [x.strip() for x in args.initializer_tokens.split(",")]
    else:
        initializer_tokens = []

    if len(initializer_tokens) == 0:
        if args.resume_from_checkpoint:
            logger.info("* Resume the embeddings of placeholder tokens *")
            print("* Resume the embeddings of placeholder tokens *")
        else:
            logger.info("* Initialize the newly added placeholder token with the random embeddings *")
            print("* Initialize the newly added placeholder token with the random embeddings *")
        token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    else:
        logger.info("* Initialize the newly added placeholder token with the embeddings of the initializer token *")
        print("* Initialize the newly added placeholder token with the embeddings of the initializer token *")
        token_ids = tokenizer.encode(initializer_tokens, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > len(initializer_tokens):
            raise ValueError("The initializer token must be a single token.")

    initializer_token_ids = token_ids
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_ids] = token_embeds[initializer_token_ids]
    if args.normalize_word:
        token_embeds[placeholder_token_ids] = F.normalize(token_embeds[placeholder_token_ids], dim=1, p=2)

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
        text_encoder.get_input_embeddings().parameters() if args.test else []
    )
    freeze_params(params_to_freeze)

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    train_dataset = ComposableDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
        placeholder_tokens=args.placeholder_tokens,
        num_images_per_class=args.num_images_per_class,
        class_folder_names=args.class_folder_names,
        learnable_property=args.learnable_property,
        set="train" if not args.test else "val",
    )

    if args.add_weight_per_score:
        # Add a learnable weight for each token
        if args.resume_from_checkpoint and args.reuse_weights:
            weight_path = os.path.join(args.resume_dir, "weights.bin")
            concept_weights = torch.load(weight_path)["weights"]
            concept_weights.requires_grad = not args.freeze_weights
            concept_weights = torch.nn.Parameter(concept_weights, requires_grad=not args.freeze_weights)
            print('reusing the weights...')
        else:
            num_tokens = len(placeholder_token_ids)
            # create weight matrix NxMx1x1x1 where D is the number of images and M is the number of classes
            concept_weights = torch.tensor([args.init_weight] * num_tokens).reshape(1, -1, 1, 1, 1).float()
            if args.softmax_weights:
                concept_weights = F.softmax(concept_weights, dim=1)
            concept_weights = concept_weights.repeat(train_dataset.num_images, 1, 1, 1, 1)
            concept_weights.requires_grad = not args.freeze_weights
            concept_weights = torch.nn.Parameter(concept_weights, requires_grad=not args.freeze_weights)

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(
            text_encoder.get_input_embeddings().parameters() if not args.test else [],
            [concept_weights] if args.add_weight_per_score else []
        ),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    args.max_train_steps = train_dataset.num_images * args.num_iters_per_image
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("checkpoints", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    print(f'total_batch_size: {total_batch_size}')

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.resume_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            args.max_train_steps = train_dataset.num_images * args.num_iters_per_image
        else:
            if not args.test:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(args.resume_dir, path))

            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
            # update the number of iterations
            args.max_train_steps = global_step + train_dataset.num_images * args.num_iters_per_image
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    # iterate through the data and save dataset info
    dataset_info = {}
    for step, batch in enumerate(train_dataloader):
        image_path = batch["image_path"]
        image_idx = batch["image_index"]
        for i in range(len(image_path)):
            dataset_info[image_idx[i].item()] = image_path[i]

    if args.test:
        path = os.path.join(args.output_dir, "test_dataset_info.json")
    else:
        path = os.path.join(args.output_dir, "dataset_info.json")

    with open(path, "w") as f:
        json.dump(dataset_info, f)

    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            mse_loss, orthogonal_loss, repulsion_loss, word_norm_loss, euclidean_dist_loss = 0, 0, 0, 0, 0
            with accelerator.accumulate(text_encoder):
                # image shape: Bx3xHxW
                # input_ids shape: BxMxD where M is the number of classes, D is the text dims
                pixel_value, input_ids = batch["pixel_values"], batch["input_ids"]
                weight_id = batch["gt_weight_id"]
                # split input ids into a list of BxD
                input_ids_list = [y.squeeze(dim=1) for y in input_ids.chunk(chunks=input_ids.shape[1], dim=1)]

                if args.use_composed_score:
                    noise, uncond_noise_pred, noise_preds = None, None, []
                    for input_ids in input_ids_list:
                        # Convert images to latent space
                        latents = vae.encode(pixel_value).latent_dist.sample().detach()
                        latents = latents * 0.18215

                        # Sample noise that we'll add to the latents
                        if noise is None:
                            noise = torch.randn(latents.shape).to(latents.device)
                        bsz = latents.shape[0]

                        # Sample a random timestep for each image
                        if args.weighted_sampling:
                            weights = torch.arange(1, noise_scheduler.config.num_train_timesteps + 1).float()
                            if args.flip_weights:
                                weights = weights.flip(dims=(0,))
                            timesteps = torch.multinomial(weights, bsz).to(latents.device)
                        else:
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                            ).long()

                        # Add noise to the latents according to the noise magnitude at each timestep
                        # (this is the forward diffusion process)
                        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                        # Get the text embedding for conditioning
                        encoder_hidden_states = text_encoder(input_ids)[0]
                        # Predict the noise residual
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        noise_preds.append(noise_pred)

                        if uncond_noise_pred is None and args.use_conj_score:
                            # precompute the unconditional text hidden states
                            uncond_text_ids = tokenizer(
                                "",
                                padding="max_length",
                                truncation=True,
                                max_length=tokenizer.model_max_length,
                                return_tensors="pt",
                            ).input_ids.to(latents.device)
                            B = noisy_latents.shape[0]
                            uncond_encoder_hidden_states = text_encoder(uncond_text_ids)[0].repeat(B, 1, 1)
                            uncond_noise_pred = unet(noisy_latents, timesteps, uncond_encoder_hidden_states).sample

                    noise_preds_stack = torch.stack(noise_preds, dim=1)  # BxMx4x64x64
                elif args.use_conj_score:
                    # latents
                    latents = vae.encode(pixel_value).latent_dist.sample().detach()
                    latents = latents * 0.18215
                    bsz = latents.shape[0]
                    noise = torch.randn_like(latents)

                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    weights = concept_weights[weight_id]

                    cond_scores = []
                    for input_ids in input_ids_list:
                        encoder_hidden_state = text_encoder(input_ids)[0].to(dtype=weight_dtype)
                        cond_scores.append(unet(noisy_latents, timesteps, encoder_hidden_state).sample)
                    cond_scores = torch.stack(cond_scores, dim=1)
                    uncond_text_ids = tokenizer(
                        "",
                        padding="max_length",
                        truncation=True,
                        max_length=tokenizer.model_max_length,
                        return_tensors="pt",
                    ).input_ids.to(latents.device)
                    uncond_encoder_hidden_states = text_encoder(uncond_text_ids)[0].repeat(bsz, 1, 1)
                    uncond_score = unet(noisy_latents, timesteps, uncond_encoder_hidden_states).sample

                    # compute initial compositional score
                    composed_score = uncond_score + torch.sum(weights.to(latents.device) * (cond_scores - uncond_score[:, None]), dim=1)
                    # encoder_hidden_states = torch.stack(encoder_hidden_states_list, dim=1)
                    # gt_encoder_states = (encoder_hidden_states * weights.to(latents.device)).sum(dim=1)
                    mse_loss = args.mse_coeff * F.mse_loss(noise, composed_score.float(), reduction="mean")

                    # orthogonal loss
                    if args.use_orthogonal_loss:
                        if args.use_similarity:
                            B, M, C, H, W = cond_scores.shape
                            ortho_scores_view = cond_scores.view(B, M, -1)
                            prod_matrix = torch.bmm(ortho_scores_view, ortho_scores_view.transpose(2, 1)) / (C * H * W)
                            # only compute the upper triangular matrices (exclude the diagonal)
                            r, c = torch.triu_indices(M, M, offset=1)
                            orthogonal_loss = args.orthogonal_coeff * (prod_matrix[:, r, c] ** 2).sum().sqrt()
                        elif args.use_euclidean_mhe:
                            B, M, C, H, W = cond_scores.shape
                            ortho_scores_view = cond_scores.view(B, M, -1)
                            batch_pair_wise_l2_dist = torch.cdist(ortho_scores_view, ortho_scores_view, p=2.0)
                            # only compute the upper triangular matrices (exclude the diagonal)
                            energy_matrix = torch.triu(batch_pair_wise_l2_dist, diagonal=1)
                            energy_matrix = energy_matrix[energy_matrix != 0]
                            if args.log_mhe:
                                orthogonal_loss = torch.log(1 / energy_matrix).mean()
                            else:
                                orthogonal_loss = (1 / energy_matrix).mean()
                            orthogonal_loss *= args.orthogonal_coeff

                    if args.text_loss:
                        word_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            placeholder_token_ids]
                        if args.text_repulsion_loss:
                            if args.use_l2_norm_regularization:
                                # word embeds are unnormalized.
                                word_norm_loss = args.l2_norm_coeff * torch.norm(word_embeds, dim=1).mean(dim=0)
                            if args.normalize_score:
                                word_embeds = F.normalize(word_embeds, p=2, dim=1)
                            word_dist_matrix = F.pdist(word_embeds, p=2)
                            repulsion_loss = args.text_repulsion_coeff * torch.log(1 / word_dist_matrix).mean()
                        elif args.text_repulsion_similarity_loss:
                            num_words = word_embeds.shape[0]
                            similarity = word_embeds @ word_embeds.T
                            similarity = similarity[torch.triu_indices(num_words, num_words, offset=1).unbind()] ** 2.
                            repulsion_loss = args.text_repulsion_coeff * similarity.sum().sqrt()

                if args.use_composed_score:
                    # extract the corresponding weights for the batch of images
                    if args.add_weight_per_score:
                        weights = concept_weights[weight_id]
                        if args.softmax_weights:
                            weights = F.softmax(concept_weights[weight_id], dim=1)
                        weighted_scores = noise_preds_stack * weights.to(latents.device)
                    else:
                        weighted_scores = noise_preds_stack / noise_preds_stack.shape[1]

                    if args.use_conj_score:
                        uncond_noise_pred = uncond_noise_pred[:, None]
                        score = uncond_noise_pred + weights.to(latents.device) * (noise_preds_stack - uncond_noise_pred)
                        composed_score = score.sum(dim=1)
                    else:
                        composed_score = weighted_scores.sum(dim=1)

                    # TODO: MSE between classifier free score and noise doesn't make sense??
                    mse_loss = args.mse_coeff * F.mse_loss(composed_score, noise, reduction="mean")
                    # compute sum of pair wise dot product as the orthogonal loss

                    if args.use_orthogonal_loss:
                        if args.use_weighted_score:
                            ortho_scores = weighted_scores
                        else:
                            ortho_scores = noise_preds_stack
                        # assume number of classes: B > 1
                        B, M, C, H, W = ortho_scores.shape
                        ortho_scores_view = ortho_scores.view(B, M, -1)
                        if args.normalize_score:
                            ortho_scores_view = F.normalize(ortho_scores_view, p=2, dim=2)

                        if args.use_similarity:
                            prod_matrix = torch.bmm(ortho_scores_view, ortho_scores_view.transpose(2, 1))
                            # only compute the upper triangular matrices (exclude the diagonal)
                            num_pairs = math.factorial(M) / (math.factorial(2) * math.factorial(M - 2))
                            ortho_matrix = (torch.triu(prod_matrix, diagonal=1) / (B * C * H * W))
                            orthogonal_loss = args.orthogonal_coeff * ortho_matrix.sum() / num_pairs
                            # r, c = torch.triu_indices(M, M, offset=1).unbind()
                            # orthogonal_loss = args.orthogonal_coeff * (prod_matrix[:, r, c]).mean()
                        elif args.use_euclidean_mhe:
                            # Minimum Hyperspherical energy: norm
                            # scale * sum_i^{N}sum_j^{N}_{i!=j} log(||w_i - w_j|| ** -
                            batch_pair_wise_l2_dist = torch.cdist(ortho_scores_view, ortho_scores_view, p=2.0)
                            # only compute the upper triangular matrices (exclude the diagonal)
                            energy_matrix = torch.triu(batch_pair_wise_l2_dist, diagonal=1)
                            energy_matrix = energy_matrix[energy_matrix != 0]
                            orthogonal_loss = torch.log(1 / energy_matrix).mean()
                            orthogonal_loss *= args.orthogonal_coeff
                        elif args.use_acos_mhe:
                            prod_matrix_1 = torch.bmm(ortho_scores_view, ortho_scores_view.transpose(2, 1))
                            energy_matrix = torch.triu(prod_matrix_1, diagonal=1)
                            energy_matrix = energy_matrix[energy_matrix != 0][..., None]
                            energy_matrix = torch.acos(energy_matrix)
                            orthogonal_loss = torch.log(1 / energy_matrix).sum(dim=energy_matrix.shape[1:]).mean(dim=0)
                            orthogonal_loss *= args.orthogonal_coeff
                        else:
                            raise NotImplementedError

                if args.text_loss:
                    word_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        placeholder_token_ids]
                    if args.text_repulsion_loss:
                        if args.use_l2_norm_regularization:
                            # word embeds are unnormalized.
                            word_norm_loss = args.l2_norm_coeff * torch.norm(word_embeds, dim=1).mean(dim=0)
                        if args.normalize_score:
                            word_embeds = F.normalize(word_embeds, p=2, dim=1)
                        word_dist_matrix = F.pdist(word_embeds, p=2)
                        repulsion_loss = args.text_repulsion_coeff * torch.log(1 / word_dist_matrix).mean()
                    elif args.text_repulsion_similarity_loss:
                        if args.use_l2_norm_regularization:
                            # word embeds are unnormalized.
                            word_norm_loss = args.l2_norm_coeff * torch.norm(word_embeds, dim=1).mean(dim=0)
                        N = word_embeds.shape[0]
                        similarity = word_embeds @ word_embeds.T
                        similarity = similarity[torch.triu_indices(N, N, offset=1).unbind()] ** 2.
                        repulsion_loss = args.text_repulsion_coeff * similarity.mean()

                if args.euclidean_dist_loss:
                    word_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        placeholder_token_ids]
                    euclidean_dist_loss = args.euclidean_dist_coeff * (1 / F.pdist(word_embeds, p=2)).mean()

                loss = mse_loss + orthogonal_loss + repulsion_loss + word_norm_loss + euclidean_dist_loss

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                if not args.test:
                    index_no_updates = torch.ones(len(tokenizer), dtype=torch.bool)
                    index_no_updates[placeholder_token_ids] = False
                    if accelerator.num_processes > 1:
                        grads = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads = text_encoder.get_input_embeddings().weight.grad
                        # optimize all newly added tokens
                    grads.data[index_no_updates, :] = grads.data[index_no_updates, :].fill_(0)

                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_progress(text_encoder, initializer_token_ids, accelerator, args)
                    if args.add_weight_per_score:
                        save_weights(concept_weights, args)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if not args.test:
                            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "mse_loss": mse_loss.item(),
                    "ortho_loss": orthogonal_loss.item() if args.use_orthogonal_loss else 0,
                    "word_repulsion_loss": repulsion_loss.item() if args.text_loss else 0,
                    "euclidean_dist_loss": euclidean_dist_loss.item() if args.euclidean_dist_loss else 0,
                    "word_norm_regularization": word_norm_loss.item() if args.use_l2_norm_regularization else 0,
                    }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.sync_gradients and global_step % args.validation_step == 0:
                folder = os.path.join(args.output_dir, f'generated_samples_{global_step}')
                os.makedirs(folder, exist_ok=True)
                logger.info(
                    f"Running validation..."
                )
                # create pipeline (note: unet and vae are loaded again in float32)
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    unet=unet,
                    vae=vae,
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                )
                pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = (
                    None if args.seed is None else
                    torch.Generator(device=accelerator.device).manual_seed(args.seed)
                )
                images = []
                prompts = []
                if args.learnable_property != "":
                    properties = [x.strip() for x in args.learnable_property.split(",")]
                else:
                    properties = []

                if properties:
                    for p, placeholder in zip(properties, placeholder_tokens):
                        if p == "object":
                            prompts.append(f"a photo of {placeholder}")
                        else:
                            prompts.append(f"a painting in the style of {placeholder}")
                else:
                    for placeholder in placeholder_tokens:
                        prompts.append(f"{placeholder}")

                for prompt in prompts:
                    image_list = pipeline(prompt, guidance_scale=7.5,
                                          num_inference_steps=50, generator=generator)
                    image = image_list.images[0]
                    image.save(os.path.join(folder, f'{prompt}.png'))
                    images.append(image)

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                del pipeline
                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

        # Create the pipeline using the trained modules and save it.
        if accelerator.is_main_process and global_step % args.checkpointing_steps == 0 and not args.test:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
            # Also save the newly trained embeddings
            save_progress(text_encoder, initializer_token_ids, accelerator, args)
            if args.add_weight_per_score:
                save_weights(concept_weights, args)

            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
