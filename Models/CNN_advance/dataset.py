import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Function to create and return data loaders for training and validation datasets
def get_data_loaders(config):
    #Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Upsample to 224x224
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load training and validation dataset from directory using ImageFolder and apply transformations
    train_dataset = datasets.ImageFolder(config['train_dir'], transform=transform)
    val_dataset = datasets.ImageFolder(config['val_dir'], transform=transform)

    # Create data loaders for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

     # Return both data loaders
    return train_loader, val_loader