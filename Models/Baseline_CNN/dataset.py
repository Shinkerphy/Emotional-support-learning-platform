# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(config['train_dir'], transform=transform)
    val_dataset = datasets.ImageFolder(config['val_dir'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, val_loader