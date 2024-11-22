import os
from torch.utils.data import Dataset, DataLoader
from preprocessing import ImagePreprocessor

from typing import Tuple, List, Optional, Union
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_path: str = 'oxford_pet', transform = None):
        self.root_dir = data_path
        self.transform = transform
        self.class_to_idx, self.idx_to_class = self._find_classes()
        self.images, self.labels = self._load_data()

    def _find_classes(self) -> Tuple[dict, dict]:
        classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        return class_to_idx, idx_to_class
    
    def find_classes(self) -> Tuple[dict, dict]:
        return self.class_to_idx, self.idx_to_class
    

    def _load_data(self) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []
        for d in os.scandir(self.root_dir):
            if d.is_dir():
                label = self.class_to_idx[d.name]
                for f in os.scandir(d.path):
                    if f.is_file():
                        image_paths.append(f.path)
                        labels.append(label)
        return image_paths, labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_dataloader(train_dataset, val_dataset, batch_size: int = 16, num_workers: int = 6):
    train_data = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_data = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_data, val_data

