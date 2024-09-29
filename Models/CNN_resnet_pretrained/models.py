import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

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

# Define CBAM: Channel Attention and Spatial Attention

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# Modify the ResNet-50 with CBAM
class EmotionResNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionResNet, self).__init__()
        
        # Load the pre-trained ResNet model
        self.resnet = models.resnet50(pretrained=True)
        
        # Adjust the first convolution layer to accept 1-channel grayscale images
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Insert CBAM into each residual block (ResNet Bottleneck Block)
        self.resnet.layer1 = self._add_cbam(self.resnet.layer1)
        self.resnet.layer2 = self._add_cbam(self.resnet.layer2)
        self.resnet.layer3 = self._add_cbam(self.resnet.layer3)
        self.resnet.layer4 = self._add_cbam(self.resnet.layer4)
        
        # Replace the last fully connected layer with one that has the desired number of output classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.BatchNorm1d(num_features),   # BatchNorm for regularization
            nn.Dropout(0.5),                # Dropout to prevent overfitting
            nn.Linear(num_features, 1024),  # Fully connected layer
            nn.ReLU(),
            nn.Dropout(0.3),                # Dropout again for additional regularization
            nn.Linear(1024, num_classes)    # Final layer for emotion classification
        )

    def _add_cbam(self, layer):
        """Utility function to add CBAM to each ResNet block."""
        for i, block in enumerate(layer):
            layer[i].cbam = CBAM(block.conv3.out_channels)
        return layer

    def forward(self, x):
        x = self.resnet(x)
        return x

# Example instantiation
# model = EmotionResNetCBAM(num_classes=7)