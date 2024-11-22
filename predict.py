import torch
from PIL import Image
import argparse
import os

from preprocessing import ImagePreprocessor
from dataloader import CustomDataset

def load_model(model_path, device):
    """Load the pre-trained model."""
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess the input image."""
    preprocessor = ImagePreprocessor().val_transform()
    image = Image.open(image_path).convert('RGB')
    image = preprocessor(image)
    return image

def predict(model, image, device):
    """Make predictions using the model."""
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)
    image = preprocess_image(args.image_path)
    prediction = predict(model, image, device)
    _, idx_to_class = CustomDataset(args.data_path).find_classes()
    prediction = idx_to_class[prediction]
    print(f'Predicted class: {prediction}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using a pre-trained model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to find name of classes')
    args = parser.parse_args()
    main(args)