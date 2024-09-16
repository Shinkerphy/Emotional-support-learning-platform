import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.leaky_relu(out, 0.1)
        return out

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

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)
        
        self.attention1 = SelfAttention(128)
        self.attention2 = SelfAttention(256)
        self.attention3 = SelfAttention(512)
        
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(256)
        self.cbam3 = CBAM(512)

        self.trans_block1 = TransformerBlock(128)
        self.trans_block2 = TransformerBlock(256)
        self.trans_block3 = TransformerBlock(512)

        self.dropout = nn.Dropout(0.5)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)), 0.1))
        
        x = self.res_block1(x)
        x = self.attention1(x)
        x = self.cbam1(x)
        x = self.trans_block1(x)
        x = self.pool(x)
        
        x = self.res_block2(x)
        x = self.attention2(x)
        x = self.cbam2(x)
        x = self.trans_block2(x)
        x = self.pool(x)
        
        x = self.res_block3(x)
        x = self.attention3(x)
        x = self.cbam3(x)
        x = self.trans_block3(x)
        x = self.pool(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.leaky_relu(self.fc1(x), 0.1))
        x = self.dropout(F.leaky_relu(self.fc2(x), 0.1))
        x = self.fc3(x)
        
        return x