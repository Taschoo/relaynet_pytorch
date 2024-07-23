import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 3), padding=(3, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x, indices = self.pool(x)
        return x, indices

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(7, 3), padding=(3, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x, indices, output_size, concat_features):
        x = self.unpool(x, indices, output_size=output_size)
        x = torch.cat((x, concat_features), dim=1)  # Concatenate along the channel dimension
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ClassificationBlock(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x
