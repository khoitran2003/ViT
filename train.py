import os
import gc
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from vit.models import ViTBase, ViTLarge, ViTHuge
from dataloader import CustomDataset, get_dataloader
from preprocessing import ImagePreprocessor
from torch.utils.tensorboard import SummaryWriter


def setup_device(seed: int = 8325):
    """Setup device and seed for reproducibility."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    return device


def initialize_model(args):
    """Initialize model based on user-defined arguments."""
    if args.model == "ViTBase":
        return ViTBase(
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            image_size=args.image_size,
        )
    elif args.model == "ViTLarge":
        return ViTLarge(
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            image_size=args.image_size,
        )
    elif args.model == "ViTHuge":
        return ViTHuge(
            num_classes=args.num_classes,
            patch_size=args.patch_size,
            image_size=args.image_size,
        )
    else:
        raise ValueError(f"Invalid model name: {args.model}")


def print_training_info(args, model, device):
    """Print training details and model parameters."""
    print("-" * 20, "Training ViT model", "-" * 20)
    print("Github: khoitran2003")
    print("Email: anhkhoi246813579@gmail.com")
    for i, arg in enumerate(vars(args)):
        print(f"{i + 1}. {arg}: {vars(args)[arg]}")
    print(f"Parameters of model: {sum(p.numel() for p in model.parameters())}")
    print(f"Device: {device}")
    print("-" * 25, "Training", "-" * 25)


def setup_data(args):
    """Setup datasets and dataloaders."""
    if not os.path.exists(args.train_folder) or not os.path.exists(args.valid_folder):
        raise FileNotFoundError("Training or validation data folder not found")

    train_transform = ImagePreprocessor().train_transform()
    val_transform = ImagePreprocessor().val_transform()

    train_dataset = CustomDataset(
        data_path=args.train_folder, transform=train_transform
    )
    val_dataset = CustomDataset(data_path=args.valid_folder, transform=val_transform)

    return get_dataloader(train_dataset, val_dataset, args.batch_size)


def train_loop(
    args,
    train_loader,
    val_loader,
    model,
    optim,
    criterion,
    start_epoch,
    best_val_loss,
    device,
    writer
):
    """Run the training loop."""
    if device.type == "cuda":
        from torch.amp import GradScaler, autocast
        scaler = GradScaler()
        print("Using CUDA for training")
    else:
        print("Using CPU for training")
    train_loss = 0
    train_acc = 0
    for epoch in range(start_epoch, args.epochs):
        train_progressbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", colour="green"
        )
        for images, labels in train_progressbar:
            optim.zero_grad()
            images, labels = images.to(device), labels.to(device)
            if device.type == "cuda":
                with autocast(device_type=device.type):
                    logits = model(images)
                    train_loss = criterion(logits, labels)
                scaler.scale(train_loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                logits = model(images)
                train_loss = criterion(logits, labels)
                train_loss.backward()
                optim.step()
            train_loss += train_loss.item()
            train_acc += (logits.argmax(1) == labels).float().mean()
            train_progressbar.set_description(
                f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}"
            )

            # Free memory by deleting mo
            del images, labels, logits
            if device.type == "cuda":
                torch.cuda.empty_cache()
            else:
                gc.collect()
    
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_acc:.4f}")
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            val_progressbar = tqdm(val_loader, desc="Validation", colour="red")
            for images, labels in val_progressbar:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                val_acc += (logits.argmax(1) == labels).float().mean()
                val_progressbar.set_description(
                    f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}"
                )
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.model_folder, args.model + "_best.pth")
            torch.save({"model_state_dict": model.state_dict()}, best_model_path)
            print(f"Best model saved at {best_model_path}")
        last_model_path = os.path.join(args.model_folder, args.model + "_last.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_state_dict": optim.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            last_model_path,
        )
        print(f"Last model saved at {last_model_path}")
    print("Training completed")


def load_pretrained_model(args, model, optim, device):
    """Load pre-trained model."""
    last_model_path = os.path.join(args.model_folder, args.model + "_last.pth")
    if os.path.exists(last_model_path):
        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"Loaded model from {last_model_path}, starting from epoch {start_epoch}")
        return start_epoch, best_val_loss
    else:
        print("No pre-trained model found")
        return 0, float("inf")


def train(args):
    """Main training function."""
    device = setup_device()

    # Initialize model
    model = initialize_model(args).to(device)

    # Optimizer
    optim = torch.optim.Adam(
        params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Tensorboard
    writer = SummaryWriter()

    # Print training information
    print_training_info(args, model, device)

    # check pre-trained model
    start_epoch, best_val_loss = load_pretrained_model(args, model, optim, device)
    if start_epoch == 0:
        print("Continuing without pre-trained model")

    # Setup data
    try:
        train_loader, val_loader = setup_data(args)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # Create model folder
    os.makedirs(args.model_folder, exist_ok=True)

    # Training loop
    train_loop(
        args,
        train_loader,
        val_loader,
        model,
        optim,
        criterion,
        start_epoch,
        best_val_loss,
        device,
        writer
    )

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="ViTBase",
        choices=["ViTBase", "ViTLarge", "ViTHuge"],
        help="Choose the model architecture",
    )
    parser.add_argument("--num-classes", default=37, type=int, help="Number of classes")
    parser.add_argument(
        "--patch-size", default=16, type=int, help="Size of image patch"
    )
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size")
    parser.add_argument(
        "--epochs", default=500, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--image-size", default=224, type=int, help="Size of input image"
    )
    parser.add_argument(
        "--train-folder",
        default="oxford_pet/train",
        type=str,
        help="Where training data is located",
    )
    parser.add_argument(
        "--valid-folder",
        default="oxford_pet/val",
        type=str,
        help="Where validation data is located",
    )
    parser.add_argument(
        "--model-folder",
        default="output/",
        type=str,
        help="Folder to save trained model",
    )
    args = parser.parse_args()

    train(args)
