import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y

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

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        self.expand_ratio = expand_ratio

        hidden_dim = in_channels * expand_ratio
        self.expand = in_channels != hidden_dim
        if self.expand:
            self.expand_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False)
            self.bn0 = nn.BatchNorm2d(hidden_dim)
            self.act = Swish()

        self.depthwise_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        self.se_block = SEBlock(hidden_dim, int(in_channels * se_ratio))
        self.self_attention = SelfAttention(hidden_dim)

        self.project_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = x

        if self.expand:
            out = self.expand_conv(out)
            out = self.bn0(out)
            out = self.act(out)

        out = self.depthwise_conv(out)
        out = self.bn1(out)
        out = self.act(out)

        out = self.se_block(out)
        out = self.self_attention(out)

        out = self.project_conv(out)
        out = self.bn2(out)

        if self.use_res_connect:
            return identity + out
        else:
            return out

class EfficientNet(nn.Module):
    def __init__(self, width_coefficient, depth_coefficient, dropout_rate, num_classes=7):
        super(EfficientNet, self).__init__()
        self.swish = Swish()
        self.stem_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(32)
        
        # Define the blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(self._make_stage(32, 16, 1, 1, width_coefficient, depth_coefficient, 1))
        self.blocks.append(self._make_stage(16, 24, 6, 2, width_coefficient, depth_coefficient, 2))
        self.blocks.append(self._make_stage(24, 40, 6, 2, width_coefficient, depth_coefficient, 2))
        self.blocks.append(self._make_stage(40, 80, 6, 2, width_coefficient, depth_coefficient, 3))
        self.blocks.append(self._make_stage(80, 112, 6, 1, width_coefficient, depth_coefficient, 3))
        self.blocks.append(self._make_stage(112, 192, 6, 2, width_coefficient, depth_coefficient, 4))
        self.blocks.append(self._make_stage(192, 320, 6, 1, width_coefficient, depth_coefficient, 1))
        
        self.head_conv = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.head_bn = nn.BatchNorm2d(1280)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1280, num_classes)

    def _make_stage(self, in_channels, out_channels, expand_ratio, stride, width_coefficient, depth_coefficient, repeats):
        layers = []
        in_channels = int(in_channels * width_coefficient)
        out_channels = int(out_channels * width_coefficient)
        repeats = int(repeats * depth_coefficient)
        layers.append(MBConvBlock(in_channels, out_channels, expand_ratio, stride))
        for _ in range(1, repeats):
            layers.append(MBConvBlock(out_channels, out_channels, expand_ratio, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.swish(x)

        for block in self.blocks:
            x = block(x)

        x = self.head_conv(x)
        x = self.head_bn(x)
        x = self.swish(x)

        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def efficientnet_b0(num_classes=7):
    return EfficientNet(1.0, 1.0, 0.2, num_classes)

def efficientnet_b1(num_classes=7):
    return EfficientNet(1.0, 1.1, 0.2, num_classes)

def efficientnet_b2(num_classes=7):
    return EfficientNet(1.1, 1.2, 0.3, num_classes)

def efficientnet_b3(num_classes=7):
    return EfficientNet(1.2, 1.4, 0.3, num_classes)

def efficientnet_b4(num_classes=7):
    return EfficientNet(1.4, 1.8, 0.4, num_classes)

def efficientnet_b5(num_classes=7):
    return EfficientNet(1.6, 2.2, 0.4, num_classes)

def efficientnet_b6(num_classes=7):
    return EfficientNet(1.8, 2.6, 0.5, num_classes)

def efficientnet_b7(num_classes=7):
    return EfficientNet(2.0, 3.1, 0.5, num_classes)
