import os
import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Custom dataset class to handle AffectNet with labels from CSV
class AffectNetDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Mapping emotions (from string) to integer labels
        self.emotion_dict = {'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 'neutral': 5, 'sad': 6, 'surprise': 7}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract the full path from the CSV
        img_path_in_csv = str(self.data.iloc[idx, 1])  # Get the image path like 'contempt/image0039811.jpg'

        # Extract the emotion category from the file path (everything before '/')
        emotion_str = img_path_in_csv.split('/')[0]  # Extract 'contempt' from 'contempt/image0039811.jpg'
        
        # Extract the image filename without the folder part
        img_name = os.path.basename(img_path_in_csv)  # Get just 'image0039811.jpg'

        # Map the emotion string to the corresponding integer
        emotion_label = self.emotion_dict[emotion_str]  # Get the integer label for the emotion

        # Define possible file extensions
        possible_extensions = ['.jpg', '.png']
        img_path = None

        # Try each extension until a valid file is found
        for ext in possible_extensions:
            temp_path = os.path.join(self.img_dir, emotion_str, img_name)  # Use both the emotion folder and image name
            if os.path.exists(temp_path):
                img_path = temp_path
                break

        if img_path is None:
            print(f"Image file not found for {img_name} in {emotion_str}")  # Log missing files
            raise FileNotFoundError(f"Image file not found for {img_name} in {emotion_str}")

        # Open the image and apply transformations
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, emotion_label

# Function to split data into training and validation sets
def get_data_loaders(config):
    # Define transformation for AffectNet (upsampling to 224x224 RGB images)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Upsample to 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB normalization
    ])

    # Load data from CSV
    dataset = AffectNetDataset(csv_file=config['csv_file'], img_dir=config['img_dir'], transform=transform)

    # Split dataset into 80% training and 20% validation
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    return train_loader, val_loader