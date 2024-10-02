import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG-like Block with multiple convolutional layers, batch normalization, and ReLU
class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

# Main CNN model for emotion classification
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # VGG-like Convolutional blocks
        self.conv1 = VGGBlock(1, 64, 2)  # First block with 2 conv layers
        self.conv2 = VGGBlock(64, 128, 2)  # Second block with 2 conv layers
        self.conv3 = VGGBlock(128, 256, 3)  # Third block with 3 conv layers
        self.conv4 = VGGBlock(256, 512, 3)  # Fourth block with 3 conv layers
        self.conv5 = VGGBlock(512, 512, 3)  # Fifth block with 3 conv layers
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = self.pool(self.conv1(x))  # Apply first conv block and pool
        x = self.pool(self.conv2(x))  # Apply second conv block and pool
        x = self.pool(self.conv3(x))  # Apply third conv block and pool
        x = self.pool(self.conv4(x))  # Apply fourth conv block and pool
        x = self.pool(self.conv5(x))  # Apply fifth conv block and pool
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout(F.relu(self.fc1(x)))  # First fully connected layer
        x = self.dropout(F.relu(self.fc2(x)))  # Second fully connected layer
        x = self.fc3(x)  # Final output layer
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# Example usage:
# model = EmotionCNN(num_classes=7)