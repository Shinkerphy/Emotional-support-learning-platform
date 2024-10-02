import os
import yaml
import torch
import torch.optim as optim
import torch_optimizer as radam_optim
from torch.nn import Module
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.cuda.amp import autocast, GradScaler  # For mixed precision training

from models import EmotionCNN
from dataset import get_data_loaders
from logger import Logger
from utils import plot_model_history, evaluate_model, plot_confusion_matrix

#Loss function
class FocalLoss(Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

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
criterion = FocalLoss(alpha=1, gamma=2, reduction='mean')
optimizer = radam_optim.RAdam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
scaler = GradScaler()  # For mixed precision training

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
    
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epoch']):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        all_labels = []
        all_preds = []
        optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():  # Mixed precision
                outputs = model(images)
                loss = criterion(outputs, labels) / config['batch_size']  # Scale loss
            
            scaler.scale(loss).backward()
            
            if (i + 1) % config['batch_size'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * config['batch_size']  # Accumulated loss
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