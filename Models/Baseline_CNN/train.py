import os
import yaml
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from models import EmotionCNN
from dataset import get_data_loaders
from logger import Logger
from utils import plot_model_history, evaluate_model

# Get the directory of the current script
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, 'config.yaml')

# Load config
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get data loaders
train_loader, val_loader = get_data_loaders(config)

# Initialize model, loss, and optimizer
model = EmotionCNN().to(device)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-6)

# Define paths for checkpoint and WandB logs
checkpoint_path = os.path.join(script_dir, 'model.pth')
wandb_dir = os.path.join(script_dir, 'wandb')

# Ensure WandB directory exists
os.makedirs(wandb_dir, exist_ok=True)

def train():
    # Initialize logger once
    logger = Logger(config['experiment_name'], project=config['project_name'], dir=wandb_dir).get_logger()
    
    train_history = {'accuracy': [], 'loss': []}
    val_history = {'accuracy': [], 'loss': []}
    
    for epoch in range(config['num_epoch']):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        all_labels = []
        all_preds = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_accuracy = val_metrics['accuracy']
        val_loss = val_metrics['avg_loss']
        precision = val_metrics['precision']
        recall = val_metrics['recall']
        f1 = val_metrics['f1_score']
        val_cm = val_metrics['confusion_matrix']
        
        train_history['accuracy'].append(train_accuracy)
        train_history['loss'].append(train_loss)
        val_history['accuracy'].append(val_accuracy)
        val_history['loss'].append(val_loss)
        
        print(f"Epoch [{epoch + 1}/{config['num_epoch']}], "
              f"Train Accuracy: {train_accuracy:.2f}%, Train Loss: {train_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, "
              f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}")
        
        logger.log({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'val_confusion_matrix': val_cm
        })
    
    torch.save(model.state_dict(), checkpoint_path)
    plot_model_history(train_history, val_history, config['num_epoch'], logger)

if __name__ == '__main__':
    train()

