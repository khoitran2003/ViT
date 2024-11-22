import cv2
import numpy as np
import torch
from torchvision import transforms

from typing import Tuple, List, Optional, Union


class ImagePreprocessor:
    """
    Image preprocessor for ViT models.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        augmentations: bool = True,
    ):
        self.target_size = target_size
        self.mean = mean
        self.std = std
        self.augmentations = augmentations
        # self.train_transform = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(self.target_size),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )

    def train_transform(self) -> transforms.Compose:
        transform_list = [
            transforms.Resize(self.target_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        if self.augmentations:
            transform_list.extend(
                [
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    ),
                    transforms.RandomRotation(degrees=15),
                ]
            )
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        return transforms.Compose(transform_list)

    def val_transform(self) -> transforms.Compose:
        transform_list = [
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ]
        return transforms.Compose(transform_list)
