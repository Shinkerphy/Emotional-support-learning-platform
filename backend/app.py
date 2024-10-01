import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import threading
from PIL import Image  # Add this import

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define the model architecture to match the saved state_dict
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 7 * 7, 1024)  # Adjusted for 224x224 input
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 7)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.pool(self.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool(self.leaky_relu(self.bn3(self.conv3(x))))
        x = self.pool(self.leaky_relu(self.bn4(self.conv4(x))))
        x = self.pool(self.leaky_relu(self.bn5(self.conv5(x))))
        x = self.dropout(x)
        x = x.view(-1, 512 * 7 * 7)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN().to(device)

# Load the weights
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# Define the emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Define image transformation
transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Upsample to 224x224
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

output_frame = None
lock = threading.Lock()

## The model Inference for AffectNet dataset model(CNN upsampled)
# class EmotionCNN(nn.Module):
#     def __init__(self):
#         super(EmotionCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # For RGB input
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.fc1 = nn.Linear(512 * 7 * 7, 1024)  # Adjusted for 224x224 input
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(256, 8)  # Updated for 8 emotion classes
#         self.leaky_relu = nn.LeakyReLU(0.1)

#     def forward(self, x):
#         x = self.pool(self.leaky_relu(self.bn1(self.conv1(x))))
#         x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))
#         x = self.pool(self.leaky_relu(self.bn3(self.conv3(x))))
#         x = self.pool(self.leaky_relu(self.bn4(self.conv4(x))))
#         x = self.pool(self.leaky_relu(self.bn5(self.conv5(x))))
#         x = self.dropout(x)
#         x = x.view(-1, 512 * 7 * 7)  # Flatten for fully connected layer
#         x = self.leaky_relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.leaky_relu(self.fc2(x))
#         x = self.fc3(x)  # No activation here as it's handled in loss calculation
#         return x

# # Initialize the model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = EmotionCNN().to(device)

# # Load the model weights
# model.load_state_dict(torch.load('model.pth', map_location=device))
# model.eval()

# # Define the new emotion dictionary
# emotion_dict = {
#     0: "Angry", 
#     1: "Contempt", 
#     2: "Disgust", 
#     3: "Fear", 
#     4: "Happy", 
#     5: "Neutral", 
#     6: "Sad", 
#     7: "Surprised"
# }

def gen():
    global output_frame, lock
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            
            # Convert the ROI to a PIL image before transforming
            roi_pil = Image.fromarray(roi_gray)
            roi = transform(roi_pil).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                prediction = model(roi)
            
            maxindex = int(torch.argmax(prediction))
            emotion = emotion_dict[maxindex]
            
            # Draw bounding box and emotion on frame
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        with lock:
            output_frame = frame.copy()
        
        # Encoding the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_emotion')
def current_emotion():
    global output_frame, lock
    with lock:
        if output_frame is None:
            return jsonify({'emotion': None})
        gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return jsonify({'emotion': 'No faces detected'})
        
        emotions = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_pil = Image.fromarray(roi_gray)
            roi = transform(roi_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = model(roi)
            
            maxindex = int(torch.argmax(prediction))
            emotion = emotion_dict[maxindex]
            emotions.append(emotion)
        
        return jsonify({'emotion': emotions[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)