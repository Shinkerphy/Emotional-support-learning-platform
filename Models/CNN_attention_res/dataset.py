import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from collections import Counter

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = datasets.ImageFolder(config['train_dir'], transform=transform)
    val_dataset = datasets.ImageFolder(config['val_dir'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Compute class weights for balancing
    class_counts = Counter(train_dataset.targets)
    class_weights = [1.0 / class_counts[i] for i in range(len(class_counts))]
    sample_weights = [class_weights[t] for t in train_dataset.targets]
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=config['num_workers'])

    return train_loader, val_loader, class_weights