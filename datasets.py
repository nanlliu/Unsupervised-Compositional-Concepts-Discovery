import os
import random
import json
import torch
import numpy as np
import os.path
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer

from typing import Any, Callable, Optional, Tuple
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

PIL_INTERPOLATION = {
    "linear": Image.Resampling.BILINEAR,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
}

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class ComposableDataset(Dataset):
    def __init__(
            self,
            data_root,
            tokenizer,
            size=512,
            repeats=100,
            interpolation="bicubic",
            flip_p=0.5,
            set="train",
            placeholder_tokens="",
            center_crop=False,
            num_images_per_class=-1,
            class_folder_names="",
            learnable_property="",
    ):
        self.data_root = [x.strip() for x in data_root.split(",")]
        self.class_folder_names = [x.strip() for x in class_folder_names.split(",")]

        self.tokenizer = tokenizer
        self.size = size
        self.placeholder_tokens = [x.strip() for x in placeholder_tokens.split(",")]
        self.placeholder_tokens_ids = tokenizer.convert_tokens_to_ids(self.placeholder_tokens)

        self.center_crop = center_crop
        self.flip_p = flip_p

        # use textual inversion template - assume objects
        self.learnable_property = (x.strip() for x in learnable_property.split(","))
        self.templates = [imagenet_templates_small if x == "object" else imagenet_style_templates_small
                          for x in self.learnable_property]
        self.use_template = learnable_property != ""

        # combine all folders into a single folder
        self.image_paths, self.classes = [], []
        total_images = max(len(self.placeholder_tokens) * num_images_per_class,
                           len(self.class_folder_names) * num_images_per_class)

        images_per_folder = total_images // len(self.class_folder_names)

        for class_id, class_name in enumerate(self.class_folder_names):
            folder = os.path.join(self.data_root[class_id], class_name)
            folder_image_paths = [os.path.join(folder, file_name) for file_name in os.listdir(folder)]
            # reduce the size of images from each category if specified
            if num_images_per_class != -1:
                train_image_path = folder_image_paths[:images_per_folder]
                test_image_path = folder_image_paths[images_per_folder:2 * images_per_folder]
                if set == "train":
                    folder_image_paths = train_image_path
                else:
                    folder_image_paths = test_image_path

            self.image_paths.extend(folder_image_paths)
            self.classes.extend([class_id] * len(folder_image_paths))

        # size is the total images from different folders
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        print("placeholder_tokens: ", self.placeholder_tokens)
        print("placeholder_tokens_ids: ", self.placeholder_tokens_ids)
        print("the number of images in this dataset: ", self.num_images)
        print("the flag for using the template: ", self.use_template)

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_images
        example = dict()
        image = Image.open(self.image_paths[idx])
        # image.save(f"{self.placeholder_tokens[self.classes[idx]]}_{i}.png")

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.use_template:
            text = [random.choice(self.templates[self.classes[idx]]).format(x) for x in self.placeholder_tokens]
        else:
            text = self.placeholder_tokens  # use token itself as the caption (unsupervised)

        # encode all classes since we will use all of them to compute composed score
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        example["gt_weight_id"] = idx
        example["image_path"] = self.image_paths[idx]
        example["image_index"] = idx
        return example


class ClassificationDataset(Dataset):
    def __init__(
            self,
            image_path_map,
            learned_weights,
            image_path_to_class,
            tokenizer,
            encoder,
            size=512,
            repeats=100,
            interpolation="bicubic",
            flip_p=0.5,
            set="train",
            placeholder_tokens="",
            center_crop=False,
            learnable_property="",
    ):
        self.image_path_map = image_path_map
        self.learned_weights = learned_weights
        self.image_path_to_class = image_path_to_class

        self.tokenizer = tokenizer
        self.encoder = encoder

        self.size = size
        self.placeholder_tokens = [x.strip() for x in placeholder_tokens.split(",")]
        self.placeholder_tokens_ids = tokenizer.convert_tokens_to_ids(self.placeholder_tokens)

        self.center_crop = center_crop
        self.flip_p = flip_p

        # use textual inversion template - assume objects
        self.learnable_property = (x.strip() for x in learnable_property.split(","))
        self.templates = [imagenet_templates_small if x == "object" else imagenet_style_templates_small
                          for x in self.learnable_property]
        self.use_template = learnable_property != ""

        # size is the total images from different folders
        self.num_images = len(self.image_path_map)
        self._length = self.num_images

        print("placeholder_tokens: ", self.placeholder_tokens)
        print("placeholder_tokens_ids: ", self.placeholder_tokens_ids)
        print("the number of images in this dataset: ", self.num_images)
        print("the flag for using the template: ", self.use_template)

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_images
        image = Image.open(self.image_path_map[idx])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        if self.use_template:
            text = [random.choice(self.templates[self.classes[idx]]).format(x) for x in self.placeholder_tokens]
        else:
            text = self.placeholder_tokens  # use token itself as the caption (unsupervised)

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        # encode all classes since we will use all of them to compute composed score
        input_ids = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        example = {}
        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        example["embeddings"] = self.encoder(input_ids)[0]
        example["weights"] = self.learned_weights[idx]
        example["class"] = self.image_path_to_class[idx]
        return example

