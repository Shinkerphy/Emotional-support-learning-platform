import os
import yaml
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from torch.cuda.amp import GradScaler, autocast

from models import EmotionCNN
from dataset import get_data_loaders
from logger import Logger
from utils import plot_model_history, evaluate_model, plot_confusion_matrix

#Check config file
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, 'config.yaml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

#Work with CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader, val_loader = get_data_loaders(config)

#CNN model
model = EmotionCNN().to(device)
#optimizers, Loss, and scheduler
criterion = CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-2)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

#Checkpoint
checkpoint_path = os.path.join(script_dir, 'model.pth')
wandb_dir = os.path.join(script_dir, 'wandb')
#make new dir if not exist
os.makedirs(wandb_dir, exist_ok=True)

def train():
    logger = Logger(config['experiment_name'], project=config['project_name'], dir=wandb_dir).get_logger()
    
    train_history = {'accuracy': [], 'loss': []}
    val_history = {'accuracy': [], 'loss': []}
    
    best_val_loss = float('inf')
    
    scaler = GradScaler()
    
    accumulation_steps = 4  # Gradient accumulation steps
    
    for epoch in range(config['num_epoch']):
        torch.cuda.empty_cache()  # Clear CUDA cache
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        all_labels = []
        all_preds = []
        
        optimizer.zero_grad()
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        #Validation metrics printing
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        val_accuracy = val_metrics['accuracy']
        val_loss = val_metrics['avg_loss']
        precision = val_metrics['precision']
        recall = val_metrics['recall']
        f1 = val_metrics['f1_score']
        
        #Train metrics
        train_history['accuracy'].append(train_accuracy)
        train_history['loss'].append(train_loss)
        val_history['accuracy'].append(val_accuracy)
        val_history['loss'].append(val_loss)
        
        print(f"Epoch [{epoch + 1}/{config['num_epoch']}], "
              f"Train Accuracy: {train_accuracy:.2f}%, Train Loss: {train_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, "
              f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1: {f1:.2f}")
        
        #WandB logging
        logger.log({
            'epoch': epoch + 1,
            'train_accuracy': train_accuracy,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    torch.save(model.state_dict(), checkpoint_path)
    plot_model_history(train_history, val_history, config['num_epoch'], logger)

    final_metrics = evaluate_model(model, val_loader, criterion, device)
    plot_confusion_matrix(final_metrics['confusion_matrix'], list(range(7)), config['experiment_name'])

if __name__ == '__main__':
    train()