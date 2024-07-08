from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the Keras model architecture
def create_keras_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    return model

# Create the model and load the weights
keras_model = create_keras_model()
keras_model.load_weights('/Users/abdulmalikshinkafi/Emotion-Recognition-App/Basic_CNN/model.h5')

import torch
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.models import load_model
import numpy as np

# Define the PyTorch model
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 7)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(-1, 128 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create an instance of the PyTorch model
pytorch_model = EmotionCNN()

# Transfer weights from Keras to PyTorch
def convert_keras_to_pytorch(keras_model, pytorch_model):
    keras_weights = keras_model.get_weights()
    pytorch_params = pytorch_model.state_dict()
    
    # Mapping Keras layers to PyTorch layers
    weight_map = {
        0: 'conv1.weight', 1: 'conv1.bias',
        2: 'conv2.weight', 3: 'conv2.bias',
        6: 'conv3.weight', 7: 'conv3.bias',
        9: 'conv4.weight', 10: 'conv4.bias',
        14: 'fc1.weight', 15: 'fc1.bias',
        17: 'fc2.weight', 18: 'fc2.bias'
    }
    
    for keras_idx, pytorch_key in weight_map.items():
        weight = torch.tensor(keras_weights[keras_idx])
        if 'weight' in pytorch_key:
            weight = weight.permute(3, 2, 0, 1)  # Convert to PyTorch format if necessary
        pytorch_params[pytorch_key].copy_(weight)
    
    return pytorch_model

pytorch_model = convert_keras_to_pytorch(keras_model, pytorch_model)

# Save the converted PyTorch model
torch.save(pytorch_model.state_dict(), 'model.pth')