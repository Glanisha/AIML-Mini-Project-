"""Datasets. Evaluation inputs use the backend's preprocessing so training and serving match."""

from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from app.inference.preprocess import IMAGE_SIZE, MEAN, STD, to_model_input

from .common import ARTIFACTS, DATA_RAW

TRAIN_CACHE_SIDE = 448  # pre-shrink originals once; augmentation crops from this


@lru_cache(maxsize=None)
def load_image(rel_path: str) -> Image.Image:
    with Image.open(DATA_RAW / rel_path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


@lru_cache(maxsize=None)
def _train_source(rel_path: str) -> Image.Image:
    im = load_image(rel_path).copy()
    im.thumbnail((TRAIN_CACHE_SIDE, TRAIN_CACHE_SIDE), Image.BILINEAR)
    return im


def load_manifest() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "manifest.csv")


train_transform = T.Compose(
    [
        T.ToImage(),
        T.RandomResizedCrop(IMAGE_SIZE, scale=(0.55, 1.0), ratio=(0.8, 1.25), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.RandomApply([T.RandomRotation(20)], p=0.5),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 1.5))], p=0.2),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN.tolist(), STD.tolist()),
    ]
)


class TrainDataset(Dataset):
    def __init__(self, paths: list[str], labels: list[int]):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return train_transform(_train_source(self.paths[i])), self.labels[i]


def eval_tensor(paths: list[str]) -> torch.Tensor:
    """Stack of deployment-preprocessed inputs, shape (N, 3, 224, 224)."""
    return torch.from_numpy(np.concatenate([to_model_input(load_image(p)) for p in paths]))
