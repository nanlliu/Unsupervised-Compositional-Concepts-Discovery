"""
modified based upon https://github.com/rinongal/textual_inversion/blob/main/evaluation/clip_eval.py
"""
import argparse
import os

import clip
import torch
from torchvision import transforms
from resnet import resnet50
from PIL import Image

from typing import List
from torchmetrics.functional import kl_divergence


class CLIPEvalutor:
    def __init__(self, target_classes_names: List, device, clip_model='ViT-B/32'):
        self.device = device
        self.model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        self.texts = [f"a photo of {class_name}" for class_name in target_classes_names]

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def get_text_features(self, text, norm: bool = True) -> torch.Tensor:
        tokens = clip.tokenize(text).to(self.device)
        text_features = self.encode_text(tokens)
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def encode_images(self, image) -> torch.Tensor:
        images = self.clip_preprocess(image).to(self.device).unsqueeze(dim=0)
        return self.model.encode_image(images)

    def evaluate(self, img_folder, threshold=0.8):
        images_path = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder)]
        with torch.no_grad():
            image_features = torch.cat([self.encode_images(Image.open(path)) for path in images_path], dim=0)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = self.get_text_features(self.texts, norm=True)
            similarity = image_features @ text_features.T

            max_vals, indices = torch.max(similarity, dim=1)
            # bincount only computes frequency for non-negative values
            counts = torch.bincount(indices, minlength=text_features.shape[0])
            coverages = []
            # when computing the coverage, we don't threshold the values
            for i, caption in enumerate(self.texts):
                # print(f"class: {caption} | "
                #       f"coverage: {counts[i] / image_features.shape[0] * 100}%")
                coverages.append(counts[i] / image_features.shape[0])
            # if the probability of predicted class is < threshold, count it as misclassified
            num_misclassified_examples = torch.sum(max_vals < threshold)
            # p = target distribution, q = modeled distribution
            p = torch.tensor([1 / len(coverages)] * len(coverages))
            q = torch.tensor(coverages)
            acc = (sum(counts) - num_misclassified_examples) / image_features.shape[0]
            kl_entropy = kl_divergence(p[None], q[None])
            print(f"{img_folder}, Avg Acc: {100 * acc.item():.2f}, KL entropy: {kl_entropy.item():.4f}")


class ResNetEvaluator:
    def __init__(self, target_classes_names: List, device):
        # replace batch norm with identity function
        self.device = device
        self.model = resnet50(pretrained=True, progress=True).to(self.device)
        # disable batch norm
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        with open('imagenet_classes.txt') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # if given class names, logits will be selected based on the given class names for evaluations
        if target_classes_names:
            self.target_labels = []
            self.target_label_names = []
            for i, label_name in enumerate(self.labels):
                for target_class in target_classes_names:
                    if target_class.lower() in label_name.lower():
                        self.target_labels.append(i)
                        self.target_label_names.append(label_name)

            assert len(self.target_labels) == len(target_classes_names), \
                "the number of found labels are not the same as the given ones"
            print(self.target_labels)
            print(self.target_label_names)
            self.target_labels = sorted(self.target_labels)
        else:
            self.target_labels = list(range(1000))

    def evaluate(self, img_folder, threshold=0.8):
        images_path = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder)]
        images = torch.stack([self.transform(Image.open(path)) for path in images_path], dim=0).to(self.device)
        # predictions
        with torch.no_grad():
            pred = self.model(images)[:, self.target_labels]
            max_vals, indices = torch.max(pred, dim=1)
            # bincount only computes frequency for non-negative values
            counts = torch.bincount(indices, minlength=len(self.target_labels))
            coverages = []
            # when computing the coverage, we don't threshold the values
            for i, index in enumerate(self.target_labels):
                # print(f"class: {self.labels[index]} | "
                #       f"coverage: {counts[i] / pred.shape[0] * 100}%")
                coverages.append((counts[i] + 1e-8) / pred.shape[0])

            # if the probability of predicted class is < threshold, count it as misclassified
            num_misclassified_examples = torch.sum(max_vals < threshold)
            # p = target distribution, q = modeled distribution
            p = torch.tensor([1 / len(coverages)] * len(coverages))
            q = torch.tensor(coverages)
            acc = (sum(counts) - num_misclassified_examples) / pred.shape[0]
            kl_entropy = kl_divergence(p[None], q[None])
            print(f"{img_folder}, Avg Acc: {100 * acc.item():.2f}, KL entropy: {kl_entropy.item():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--model", type=str, choices=["textual_inversion", "ours"])
    parser.add_argument("--evaluation_metric", choices=["clip", 'resnet'])
    parser.add_argument("--class_names", type=str, nargs="+")
    parser.add_argument("--logit_threshold", type=float)
    args = parser.parse_args()

    image_folder = os.path.join(args.model_path, "samples")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.evaluation_metric == "clip":
        # build a ground truth captions for clip score
        evaluator = CLIPEvalutor(args.class_names, device=device, clip_model='ViT-B/32')
        # create folder where generated images are saved
        save_img_folder = os.path.join(args.model_path, "samples")
        evaluator.evaluate(image_folder, threshold=args.logit_threshold)
    else:
        evaluator = ResNetEvaluator(args.class_names, device=device)
        evaluator.evaluate(image_folder, threshold=args.logit_threshold)
