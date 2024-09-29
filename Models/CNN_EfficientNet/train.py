import os
import yaml
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from models import custom_efficientnet  # Import the new architecture
from dataset import get_data_loaders
from logger import Logger
from utils import plot_model_history, evaluate_model, plot_confusion_matrix

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
model = custom_efficientnet(num_classes=7).to(device)  # Use EfficientNet

criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

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
    
    # Initialize best_val_loss to a very high value
    best_val_loss = float('inf')
    
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
            'f1_score': f1
        })
        
        # Step the scheduler
        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    torch.save(model.state_dict(), checkpoint_path)
    plot_model_history(train_history, val_history, config['num_epoch'], logger)

    # Final evaluation and plot confusion matrix
    final_metrics = evaluate_model(model, val_loader, criterion, device)
    plot_confusion_matrix(final_metrics['confusion_matrix'], list(range(7)), config['experiment_name'])

    print(final_metrics['classification_report'])
    logger.log({
        'roc_auc': final_metrics['roc_auc'],
        'average_precision': final_metrics['average_precision']
    })

if __name__ == '__main__':
    train()