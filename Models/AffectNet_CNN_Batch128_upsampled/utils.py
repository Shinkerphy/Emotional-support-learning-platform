import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
import wandb

def plot_model_history(train_history, val_history, epochs, logger):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(range(1, epochs + 1), train_history['accuracy'])
    axs[0].plot(range(1, epochs + 1), val_history['accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['train', 'val'], loc='best')

    axs[1].plot(range(1, epochs + 1), train_history['loss'])
    axs[1].plot(range(1, epochs + 1), val_history['loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['train', 'val'], loc='best')

    fig.savefig('plot.png')
    plt.show()

    # Correctly log the plot using wandb.Image
    wandb.log({
        'train_accuracy_plot': wandb.Image(fig),
        'train_loss_plot': wandb.Image(fig)
    })

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    emotion_dict = {0: "Angry", 1: "Contempt", 2: "Disgust", 3: "Fear", 4: "Happy", 5: "Neutral", 6: "Sad", 7: "Surprised"}

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(data_loader)

    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    avg_precision = average_precision_score(all_labels, all_probs, average='weighted')

    return {
        'accuracy': accuracy,
        'avg_loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': classification_report(all_labels, all_preds, target_names=list(emotion_dict.values()), zero_division=0),
        'roc_auc': roc_auc,
        'average_precision': avg_precision
    }

def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.savefig('confusion_matrix.png')
    plt.show()