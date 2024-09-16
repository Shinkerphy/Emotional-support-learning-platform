import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return self.gamma * out + x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads=4, ff_hidden_dim=512):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(in_channels)
        self.ff = nn.Sequential(
            nn.Linear(in_channels, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, in_channels)
        )

    def forward(self, x):
        batch_size, C, H, W = x.size()
        x = x.view(batch_size, C, H * W).permute(2, 0, 1)  # Reshape to (N, S, E)
        x = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0] + x
        x = self.ff(self.norm2(x)) + x
        x = x.permute(1, 2, 0).view(batch_size, C, H, W)
        return x

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.vgg_block1 = VGGBlock(1, 64, 2)
        self.vgg_block2 = VGGBlock(64, 128, 2)
        self.vgg_block3 = VGGBlock(128, 256, 3)
        self.vgg_block4 = VGGBlock(256, 512, 3)
        self.vgg_block5 = VGGBlock(512, 512, 3)
        
        self.self_attention1 = SelfAttention(128)
        self.cbam1 = CBAM(128)
        self.transformer1 = TransformerBlock(128)
        
        self.self_attention2 = SelfAttention(256)
        self.cbam2 = CBAM(256)
        self.transformer2 = TransformerBlock(256)
        
        self.self_attention3 = SelfAttention(512)
        self.cbam3 = CBAM(512)
        self.transformer3 = TransformerBlock(512)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.vgg_block1(x)
        x = self.vgg_block2(x)
        x = checkpoint(self.self_attention1, x)
        x = checkpoint(self.cbam1, x)
        x = checkpoint(self.transformer1, x)
        
        x = self.vgg_block3(x)
        x = checkpoint(self.self_attention2, x)
        x = checkpoint(self.cbam2, x)
        x = checkpoint(self.transformer2, x)
        
        x = self.vgg_block4(x)
        x = self.vgg_block5(x)
        x = checkpoint(self.self_attention3, x)
        x = checkpoint(self.cbam3, x)
        x = checkpoint(self.transformer3, x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x