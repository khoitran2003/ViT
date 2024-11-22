from dataloader import CustomDataset, get_dataloader
from preprocessing import ImagePreprocessor
import os

def test():
    root_dir = "oxford_pet"
    train_transform = ImagePreprocessor().train_transform()
    val_transform = ImagePreprocessor().val_transform()

    train_dataset = CustomDataset(data_path=os.path.join(root_dir, 'train'), transform=train_transform)
    val_dataset = CustomDataset(data_path=os.path.join(root_dir, 'val'), transform=val_transform)

    train_loader, val_loader = get_dataloader(train_dataset, val_dataset)
    for images, labels in train_loader:
        print(images.shape, labels)
        break


if __name__ == "__main__":
    test = test()
