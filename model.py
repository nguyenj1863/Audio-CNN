import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A basic residual block for 2D CNNs.
    Structure:
        Conv2d (3x3) -> BatchNorm2d -> ReLU -> Conv2d (3x3) -> BatchNorm2d
        + (Optional shortcut: Conv2d (1x1) + BatchNorm2d if shape changes)
        Output = ReLU(main path + shortcut)
    Input:  Tensor (batch_size, in_channels, H, W)
    Output: Tensor (batch_size, out_channels, H_out, W_out)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=3, stride=stride, 
                                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                                kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        self.use_shortcut = (stride != 1) or (in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # x: (batch_size, in_channels, H, W)
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x) if self.use_shortcut else x
        out_add = out + shortcut
        out = torch.relu(out_add)
        # out: (batch_size, out_channels, H_out, W_out)
        return out
    
class AudioCNN(nn.Module):
    """
    Deep CNN for audio classification using residual blocks.
    Structure:
        Conv2d (7x7) -> BatchNorm2d -> ReLU -> MaxPool2d
        -> [ResidualBlock x3]
        -> [ResidualBlock x4, increases channels]
        -> [ResidualBlock x6, increases channels]
        -> [ResidualBlock x3, increases channels]
        -> AdaptiveAvgPool2d -> Dropout -> Linear (to num_classes)
    Input:  Tensor (batch_size, 1, height, width) - a mel spectrogram
    Output: Tensor (batch_size, num_classes) - logits for each class
    """
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = nn.ModuleList([ResidualBlock(in_channels=64, out_channels=64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(in_channels=64 if i == 0 else 128, out_channels=128) for i in range(4)])
        self.layer3 = nn.ModuleList(
            [ResidualBlock(in_channels=128 if i == 0 else 256, out_channels=256) for i in range(6)])
        self.layer4 = nn.ModuleList(
            [ResidualBlock(in_channels=256 if i == 0 else 512, out_channels=512) for i in range(3)])
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        # x: (batch_size, 1, height, width)
        x = self.conv1(x)
        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        # Output: (batch_size, num_classes)
        return x
