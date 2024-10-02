import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision.transforms as transforms
from models import EmotionCNN
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score
import cv2

# Function to plot training and validation accuracy and loss history
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

    logger.log({
        'train_accuracy_plot': wandb.Image(fig),
        'train_loss_plot': wandb.Image(fig)
    })

# Function to evaluate the model performance
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

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
    
# Function to plot the confusion matrix
def plot_confusion_matrix(cm, class_names, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.savefig('confusion_matrix.png')
    plt.show()

# Function to display real-time emotion prediction using webcam feed
def display():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionCNN().to(device)
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = cv2.resize(roi_gray, (224, 224))  # Adjust size if needed
            cropped_img = cropped_img.reshape(1, 1, 224, 224)
            cropped_img = torch.tensor(cropped_img, dtype=torch.float32).to(device)
            prediction = model(cropped_img)
            maxindex = int(torch.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()