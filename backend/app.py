import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import threading

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define the model architecture to match the saved state_dict
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 emotions
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
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
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

output_frame = None
lock = threading.Lock()

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
            
            # Preprocess the ROI
            roi = transform(roi_gray).unsqueeze(0).to(device)
            
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
            roi = transform(roi_gray).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = model(roi)
            
            maxindex = int(torch.argmax(prediction))
            emotion = emotion_dict[maxindex]
            emotions.append(emotion)
        
        return jsonify({'emotion': emotions[0]})

if __name__ == '__main__':
    app.run(debug=True, port=5001, threaded=True)